"""Paired comparison without silently dropping failures or changing conditions."""
from pathlib import Path
import json
import re
from course.runtime import write_json
from course.source_binding import file_sha256, validate_source_binding

EXPECTED_PROTOCOL = {
    'stand': (60., [[0., 60., 0.]]),
    'walk': (30., [[0., 30., .2]]),
    'transition': (40., [[0., 10., 0.], [10., 30., .2], [30., 40., 0.]])}

def report_path(path):
    path=Path(path)
    return path/'evaluation.json' if path.is_dir() else path

def read(path):
    return json.loads(report_path(path).read_text(encoding='utf-8'))

def source_binding(path, report):
    """Old reports may use a sidecar bound to the exact unchanged report bytes."""
    if 'source_binding' in report:
        return validate_source_binding(report['source_binding']), 'embedded'
    path=report_path(path)
    sidecar=path.parent/'source_binding.json'
    if not sidecar.is_file():
        return None, 'missing'
    value=json.loads(sidecar.read_text(encoding='utf-8'))
    if value.get('evaluation_sha256')!=file_sha256(path):
        raise ValueError(f'{path}: source-binding sidecar belongs to different report bytes')
    if value.get('schema')!='standing_course_legacy_binding_v1' or value.get('post_run_capture') is not True:
        raise ValueError(f'{path}: legacy sidecar must explicitly record retrospective capture')
    hashes={mode.get('evaluator_sha256') for mode in report.get('modes',{}).values()}
    if hashes!={value.get('evaluator_verified_sha256')} or not re.fullmatch(r'[0-9a-f]{64}',str(value.get('evaluator_verified_sha256',''))):
        raise ValueError(f'{path}: retrospective evaluator identity does not match report')
    return validate_source_binding(value.get('source_binding')), 'legacy_sidecar'

def full_protocol_issues(report, binding):
    """Recheck real durations, schedules and identities instead of trusting a flag."""
    issues=[]
    conditions=report.get('conditions')
    if not isinstance(conditions,int) or isinstance(conditions,bool) or conditions<5:
        issues.append('fewer than five conditions')
    if report.get('complete_classroom_protocol') is not True:
        issues.append('complete_classroom_protocol is not true')
    if set(report.get('modes',{}))!=set(EXPECTED_PROTOCOL):
        issues.append('all three protocol modes are required')
    for mode,(seconds,schedule) in EXPECTED_PROTOCOL.items():
        value=report.get('modes',{}).get(mode)
        if value is None:
            continue
        expected={'mode':mode,'engine':'MotrixLab/MotrixSim','seed':report.get('protocol_seed'),
                  'conditions':conditions,'requested_seconds':seconds,'schedule':schedule,
                  'steady_start_seconds':5.,'control_dt':.02,'physics_dt':.001}
        for key,target in expected.items():
            if value.get(key)!=target:
                issues.append(f'{mode}: {key} does not match classroom protocol')
        if len(value.get('rows',[]))!=conditions:
            issues.append(f'{mode}: row count does not match conditions')
        initial=value.get('initial_conditions',{})
        for key in ('position','q','velocity_world','friction'):
            if len(initial.get(key,[]))!=conditions:
                issues.append(f'{mode}: incomplete initial conditions {key}')
        if not re.fullmatch(r'[0-9a-f]{64}', str(initial.get('initial_policy_input_sha256',''))):
            issues.append(f'{mode}: missing initial policy-input identity')
        if not re.fullmatch(r'[0-9a-f]{64}',str(value.get('evaluator_sha256',''))):
            issues.append(f'{mode}: missing evaluator identity')
    if binding is None:
        issues.append('source-file binding is missing')
    return issues

def compare_results(a,b,c,output):
    paths={'A':a,'B':b,'C':c}
    reports={name:read(path) for name,path in paths.items()}
    bindings={name:source_binding(paths[name],value) for name,value in reports.items()}
    available=[value[0] for value in bindings.values() if value[0] is not None]
    if available and any(value['canonical_sha256']!=available[0]['canonical_sha256'] or value['files']!=available[0]['files'] or value.get('vendor_versions')!=available[0].get('vendor_versions') for value in available):
        raise ValueError('A/B/C source files or vendored framework versions differ')
    reference=reports['A']
    minimum_comparison_speed=1e-4
    issues={name:full_protocol_issues(value,bindings[name][0]) for name,value in reports.items()}
    full_protocol=all(not value for value in issues.values())
    for name,value in reports.items():
        if value['protocol_seed']!=reference['protocol_seed'] or value['conditions']!=reference['conditions'] or set(value['modes'])!=set(reference['modes']):
            raise ValueError(f'{name}: evaluation protocols differ')
        for mode,base in reference['modes'].items():
            current=value['modes'][mode]
            if len(current['rows'])!=value['conditions'] or len(current['rows'])!=len(base['rows']):
                raise ValueError(f'{name} {mode}: condition row count differs')
            if [r['condition'] for r in current['rows']]!=[r['condition'] for r in base['rows']]:
                raise ValueError(f'{name} {mode}: condition labels differ')
            for key in ('mode','engine','seed','conditions','requested_seconds','schedule','steady_start_seconds','control_dt','physics_dt','initial_conditions','evaluator_sha256','runtime_versions'):
                if current.get(key)!=base.get(key): raise ValueError(f'{name} {mode}: paired field differs: {key}')
        evaluator_ids={value['modes'][mode].get('evaluator_sha256') for mode in value['modes']}
        if len(evaluator_ids)>1:
            raise ValueError(f'{name}: modes were measured by different evaluator versions')
    comparisons={}
    for mode in reference['modes']:
        comparisons[mode]=[]
        for index,base in enumerate(reference['modes'][mode]['rows']):
            row={'condition':base['condition'],'A':base,'B':reports['B']['modes'][mode]['rows'][index],
                 'C':reports['C']['modes'][mode]['rows'][index]}
            for other in ('B','C'):
                denominator=base['planar_speed_steady_mean']
                value=row[other]['planar_speed_steady_mean']
                row[other+'_planar_speed_improvement_vs_A']=((denominator-value)/denominator if denominator is not None and denominator>=minimum_comparison_speed and value is not None and base['survived'] and row[other]['survived'] else None)
            denominator=row['B']['planar_speed_steady_mean']
            value=row['C']['planar_speed_steady_mean']
            row['C_planar_speed_improvement_vs_B']=((denominator-value)/denominator if denominator is not None and denominator>=minimum_comparison_speed and value is not None and row['B']['survived'] and row['C']['survived'] else None)
            comparisons[mode].append(row)
    result={'schema':'standing_course_paired_abc_v1','matched_conditions':True,
            'complete_classroom_protocol':full_protocol,'comparison_scope':'full_protocol' if full_protocol else 'smoke_only',
            'protocol_validation':issues,
            'source_bindings':{name:{'origin':origin,'canonical_sha256':binding['canonical_sha256'] if binding else None,
                                    'binding_capture':'in_report' if origin=='embedded' else 'retrospective' if origin=='legacy_sidecar' else 'missing',
                                    'files':len(binding['files']) if binding else 0} for name,(binding,origin) in bindings.items()},
            'near_zero_denominator_guard_m_s':minimum_comparison_speed,
            'models':{name:value['policy'] for name,value in reports.items()},'paired_rows':comparisons,
            'interpretation':'Failures remain failures. Percentage improvement requires both initial episodes to survive and reference speed >=1e-4m/s. Near-zero differences are not an effective improvement claim. Smoke results cannot support full classroom acceptance.',
            'hardware_tested':False}
    write_json(output,result)
    markdown=['# A / B / C 配对评估','', '评估范围：'+('完整课堂协议' if full_protocol else '仅 smoke，不能作为完整课堂验收'),'','|模式/条件|A 水平速度模均值|B|C|C相对A改善|C相对B改善|A/B/C存活|','|---|---:|---:|---:|---:|---:|---|']
    def fmt(value): return '无完整稳态' if value is None else f'{value:.6f}'
    for mode,rows in comparisons.items():
        for row in rows:
            improved=row['C_planar_speed_improvement_vs_A']
            percentage='不可比较' if improved is None else f'{improved:.1%}'
            improved_b=row['C_planar_speed_improvement_vs_B']
            percentage_b='不可比较' if improved_b is None else f'{improved_b:.1%}'
            markdown.append(f"|{mode}/{row['condition']}|{fmt(row['A']['planar_speed_steady_mean'])}|{fmt(row['B']['planar_speed_steady_mean'])}|{fmt(row['C']['planar_speed_steady_mean'])}|{percentage}|{percentage_b}|{row['A']['survived']}/{row['B']['survived']}/{row['C']['survived']}|")
    markdown+=['','速度单位m/s。参考速度低于1e-4m/s时不计算相对改善；此时应报告绝对差和测量分辨率。走路阶段的水平速度越低不代表越好；主指标只用于站立。详细保护、漂移、姿态和行走回归见JSON。']
    Path(output).with_suffix('.md').write_text('\n'.join(markdown),encoding='utf-8')
    return result
