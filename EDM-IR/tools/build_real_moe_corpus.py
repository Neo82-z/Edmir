#!/usr/bin/env python3
"""Create the owned, controlled text corpus; no tokenizer/model/network access."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edmir.benchmark.real_moe import digest, save, validate_corpus, schedule


def content(case, category):
    domains=("图书借阅", "仓库补货", "公交接驳", "设备维护", "课程预约", "档案整理")
    english=("library loans", "warehouse replenishment", "bus connections", "equipment maintenance", "course bookings", "archive processing")
    topic=case%6
    if category=="zh":
        intro=(f"请阅读下面关于{domains[topic]}的工作记录，指出最需要先核实的两项异常，区分记录中的事实与推测。"
               "请给出下一班工作人员可以执行的检查顺序，并说明哪种新证据会使你改变判断。不要编造记录以外的结论。\n")
        parts=[]
        for j in range(90):
            base=70+(case*13+j*17)%160;done=base-(j+case)%19
            parts.append(f"\n记录 {j+1}，地点 {chr(65+j%8)} 区，班次 {1+j%3}。计划处理 {base} 项，完成 {done} 项，"
                f"移交下一班 {base-done} 项；检查员抽查 {5+j%11} 项，发现 {j%4} 项缺少签字。"
                f"记录时间为第 {1+j//3} 日 {8+j%10:02d}:{(case+j*7)%60:02d}，附件编号为 {case:03d}-{j:03d}。"
                "现场人员说明，完成数来自当班系统，移交数来自纸质清单，两份记录的统计截止时间可能不同。"
                "异常项目尚未逐一核对，暂时不能将差额全部解释为丢失或延误。负责人要求保留原始单据，"
                "先检查重复编号、撤销项目与跨班移交，再向相关人员核对。另一位同事提出改变交接顺序，"
                "但尚无前后对照，因此本记录仅保存建议，不把它写成已经有效的措施。"
                f"本次新增限制是同一区域最多同时处理 {2+j%5} 项，相邻区域的空闲人员可在核实资质后支援。"
                "如果后续发现系统时钟不同步，需要重新建立共同的时间范围后比较，不应直接删除不一致的记录。\n")
    elif category=="en":
        intro=(f"Read these operational notes about {english[topic]}. Identify two discrepancies worth checking first, "
               "separate observations from explanations, and propose a short verification sequence. State what evidence "
               "would change your conclusion. Do not assume that an untested intervention was successful.\n")
        parts=[]
        for j in range(75):
            planned=90+(case*11+j*23)%190;finished=planned-(case+j)%23
            parts.append(f"\nEntry {j+1}, site {chr(65+j%8)}, shift {1+j%3}, case {case:03d}-{j:03d}. "
                f"The team planned {planned} items and marked {finished} complete. The remaining {planned-finished} "
                f"items were assigned to the following shift. An independent check covered {6+j%13} records and "
                f"found {j%5} missing approvals. The electronic report was exported at {8+j%10:02d}:{(case+j*7)%60:02d}. "
                "The handwritten handover sheet was signed later, so its reporting interval may differ. "
                "The supervisor has retained both versions and has not classified the difference as lost work. "
                "A useful first check is to compare identifiers, cancellation records, and the exact cutoff time. "
                "A colleague suggested changing the assignment order, but no controlled comparison has been completed. "
                "This suggestion should remain a hypothesis. Teams must also distinguish a missing signature from an "
                "incomplete job: the two conditions require different follow-up actions. "
                f"The temporary rule permits {2+j%6} simultaneous items in this area and one supervised handover at a time. "
                "Any later correction must retain the original value, the supporting evidence, and the person making "
                "the correction. If clocks disagree, reconstruct the interval before comparing productivity. "
                "The next shift should verify the unresolved items rather than assuming the previous summary is accurate.\n")
    else:
        intro=("Review the Python snippets and their accompanying requirements. Find two concrete correctness risks, "
               "give a small input that exposes each one, and propose a minimal repair. Discuss missing values, "
               "duplicates and the meaning of a successful operation; do not claim to have executed this code.\n")
        parts=[]
        for j in range(90):
            parts.append(f'''\n# Module {case:03d}_{j:03d}: batch reconciliation for region {j%8}.
# Requirement: input is retained; duplicate IDs should be reported rather than silently replaced.
# A completed record needs both a positive quantity and a confirmation timestamp.
# Missing confirmation is different from a confirmed quantity of zero.
def reconcile_{j}(records, limit={20+(j+case)%37}):
    by_id = {{}}
    for row in records:
        key = row.get("id")
        by_id[key] = row
    accepted = []
    waiting = []
    for key, row in by_id.items():
        quantity = row.get("quantity", 0)
        confirmed = row.get("confirmed_at")
        if quantity and confirmed:
            accepted.append((key, quantity))
        else:
            waiting.append(key)
    return {{"accepted": accepted[:limit], "waiting": waiting,
            "reported_total": sum(q for _, q in accepted)}}

# The report consumer expects reported_total to describe the returned accepted list.
# Ordering must be explained when the limit excludes otherwise valid records.
# A record with an absent ID must not overwrite another absent-ID record.
# Retrying this operation should not mutate the caller's objects.
example_{j} = [{{"id": "r{case}-{j}", "quantity": {1+j%9}, "confirmed_at": None}},
             {{"id": "r{case}-{j}", "quantity": 0, "confirmed_at": "day-{1+j//3}"}}]
''')
    return intro+"".join(parts)


def build():
    rows=[dict(id=f"case-{i:03d}",category=("zh","en","code")[i%3],
               content=content(i,("zh","en","code")[i%3])) for i in range(128)]
    result=dict(schema="edmir.real-moe-corpus.v1",origin="procedurally_authored_controlled_text",
        source="Original task texts and records authored for this experiment; no scraped/private user data",
        license="CC0-1.0",representativeness="controlled task families, not production traffic or a quality benchmark",
        requests=rows,request_content_sha256=digest(rows),token_ids_frozen=False,
        selection_rule="first 32 cases for C1; all 128 for C32; repeated blocks reuse identical texts",
        confound="request count/content subset and fill-drain effects also differ between C1 and C32",
        template_policy="GLM native template; enable_thinking=False; instruction precedes truncated context")
    validate_corpus(result)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise SystemExit("refuse to overwrite frozen corpus")
    a.out.mkdir(parents=True)
    corpus=build();save(a.out/"requests-source.json",corpus)
    save(a.out/"schedule.json",dict(schema="edmir.real-moe-schedule.v1",blocks=schedule(),
        request_content_sha256=corpus["request_content_sha256"],token_ids_frozen=False))
    print(corpus["request_content_sha256"])
