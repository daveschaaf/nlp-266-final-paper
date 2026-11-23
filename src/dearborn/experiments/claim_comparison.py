"""Experimental direct comparison gate. Unit interpretation remains model supplied."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import re

VERSION = 'claim-evidence-comparison-v1'
FIELDS = ('subject', 'metric', 'scope', 'period', 'qualifiers')
INSTRUCTION = '''
Before assigning a verdict, explicitly compare the claim with the filing fact.
Add a comparison object to EVERY annotation:
  claim_statement: copy the supplied claim statement exactly;
  subject, metric, scope, period, qualifiers: each an object with
    claim (string), evidence (string), assessment (match, mismatch, or unclear).
Do not infer an unresolved reference such as "those proceeds" from retrieval.
A numerical match does not establish matching business scope or periods.
If a material comparison is mismatch or unclear, do not mark supported.
An unrelated fact is insufficient evidence, not a contradiction.
For quantitative claims add numeric_comparison with claim and evidence objects,
each containing value, unit, scale. unit is a canonical string such as USD,
percent, basis_points, or count. scale is one, thousand, million, or billion.
Use numeric strings with no currency symbols. Infer units and scale only from
explicit source context. Evidence numeric value must occur in a source_quote
of an extracted fact, unless it is the result of the declared calculation.
For nonquantitative claims set numeric_comparison=null.
Example shape only: "numeric_comparison": {"claim":{"value":"3.3",
"unit":"USD","scale":"billion"},"evidence":{"value":"3345.1",
"unit":"USD","scale":"million"}}.
Use calculation=null when the passage directly reports the required number.
Never calculate a difference between a claim number and an evidence number.
Preserve claim numbers: 2.1% must not become 1.6%. Preserve qualifiers such as
late, reported, expected, global, and segment names in the comparison.
Quotes must be exact source substrings. Use short contiguous quotes without
ellipsis. Do not rewrite wording, remove formatting, or quote from the claim
as if it were filing evidence. Return JSON only.
'''


def number(value):
    if not isinstance(value, str) or not re.fullmatch(r'-?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?', value):
        raise ValueError('Invalid decimal number.')
    result = Decimal(value.replace(',', ''))
    if not result.is_finite():
        raise ValueError('Non-finite number.')
    return result


def numeric_match(numeric, annotation, statement):
    scales={'one':Decimal(1),'thousand':Decimal(1000),'million':Decimal(10)**6,'billion':Decimal(10)**9}
    claim,evidence=numeric['claim'],numeric['evidence']
    left,right=number(claim['value']),number(evidence['value'])
    # Bind values to the actual claim and cited text. This does not verify semantics.
    values=lambda text: {number(m) for m in re.findall(r'(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?',text)}
    if left not in values(statement):
        raise ValueError('Declared claim value is absent from the claim.')
    calc=annotation.get('executed_calculation')
    if calc:
        if right != Decimal(calc['result']):
            raise ValueError('Evidence value differs from executed calculation.')
    elif not any(right in values(f['source_quote']) for f in annotation['extracted_facts']):
        raise ValueError('Evidence value absent from cited facts.')
    if claim['scale'] not in scales or evidence['scale'] not in scales:
        return None
    units={'USD':('USD',Decimal(1)),'percent':('ratio',Decimal('0.01')),
           'basis_points':('ratio',Decimal('0.0001')),'count':('count',Decimal(1))}
    if claim['unit'] not in units or evidence['unit'] not in units:
        return None
    lc,lf=units[claim['unit']];rc,rf=units[evidence['unit']]
    if lc!=rc:
        return False
    converted=right*scales[evidence['scale']]*rf/(scales[claim['scale']]*lf)
    precision=Decimal(1).scaleb(left.as_tuple().exponent)
    return converted.quantize(precision,rounding=ROUND_HALF_UP)==left


def apply_gate(annotation, claim, numeric_enabled=True):
    comp=annotation.get('comparison')
    if comp is None and all(k in annotation for k in ('claim_statement', *FIELDS)):
        comp={k:annotation[k] for k in ('claim_statement', *FIELDS)}
        annotation['comparison']=comp
        annotation['schema_normalization']='Grouped top-level comparison fields; source values unchanged.'
    if not isinstance(comp,dict) or comp.get('claim_statement')!=claim['statement']:
        raise ValueError('Comparison must preserve original claim statement.')
    statuses=[]
    for field in FIELDS:
        item=comp.get(field)
        if not isinstance(item,dict) or any(not isinstance(item.get(k),str) for k in ('claim','evidence','assessment')) or item['assessment'] not in {'match','mismatch','unclear'}:
            raise ValueError(f'Invalid comparison field: {field}.')
        statuses.append(item['assessment'])
    if not numeric_enabled:
        annotation['model_verdict']=annotation['verdict']
        annotation['numeric_gate_enabled']=False
        if annotation['verdict']=='supported' and any(x!='match' for x in statuses):
            annotation['verdict']='insufficient_evidence'
            annotation['gate_reason']='Material comparison unresolved or mismatched.'
        return annotation
    if 'numeric_comparison' not in annotation:
        raise ValueError('Missing numeric_comparison field.')
    numeric=annotation['numeric_comparison']
    result=None
    if numeric is not None:
        try:
            result=numeric_match(numeric,annotation,claim['statement'])
        except (KeyError,TypeError,InvalidOperation) as error:
            raise ValueError('Invalid numeric comparison.') from error
    elif re.search(r'\d',claim['statement']) and annotation['verdict']=='supported':
        raise ValueError('Quantitative supported claim needs numeric comparison.')
    annotation['model_verdict']=annotation['verdict']
    annotation['numeric_gate']=result
    if annotation['verdict']=='supported' and (any(x!='match' for x in statuses) or (numeric is not None and result is not True)):
        annotation['verdict']='insufficient_evidence'
        annotation['gate_reason']='Material comparison unresolved/mismatched or numeric comparison failed.'
    return annotation
