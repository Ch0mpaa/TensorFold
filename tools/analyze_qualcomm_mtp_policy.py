#!/usr/bin/env python3
"""Replay Qualcomm MTP depth policies over teacher-forced serial-position dumps."""
import argparse, json, math, pathlib, statistics


def events(path):
    out=[]
    for line in pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("{"): continue
        try: out.append(json.loads(line))
        except json.JSONDecodeError: pass
    return out


def load(path, depth):
    raw=events(path); summary=next(x for x in reversed(raw) if x.get("tensorfold")=="qualcomm-exl3-speculative-summary")
    drafts={}; rows={}; verifies={}
    for event in raw:
        kind=event.get("tensorfold"); round_id=event.get("round")
        if kind=="qualcomm-exl3-mtp-draft": drafts[(round_id,event["draft_index"])]=event
        elif kind=="qualcomm-exl3-verify-row": rows[(round_id,event["row"])]=event
        elif kind=="qualcomm-exl3-verify-batch": verifies[round_id]=event
    records=[]
    for round_id in sorted(verifies):
        first=drafts.get((round_id,0)); row0=rows[(round_id,0)]
        second=drafts.get((round_id,1)); row1=rows.get((round_id,1))
        records.append({"position":round_id,"available_depth":0 if first is None else depth,
                        "first_margin":float(first["margin"]) if first else -math.inf,
                        "second_margin":float(second["margin"]) if second else None,
                        "first_match":bool(row0["proposal_match"]) if first else False,
                        "second_match":bool(row1["proposal_match"]) if row1 else False,
                        "first_token":int(first["proposed_token"]) if first else None,
                        "second_token":int(second["proposed_token"]) if second else None,
                        "target_token":int(row0["target_token"]),
                        "verify_ms":float(verifies[round_id]["elapsed_ms"])})
    if len(records)!=summary["rounds"]: raise ValueError(f"{path}: {len(records)} records, {summary['rounds']} rounds")
    round_ms=(float(summary["speculative_elapsed_ms"])-float(summary["mtp_prefill_ms"]))/len(records)
    return {"depth":depth,"records":records,"summary":summary,"round_ms":round_ms,
            "verify_ms_median":statistics.median(x["verify_ms"] for x in records)}


def replay(records,cost1,cost2,chooser):
    position=0; rounds=0; accepted=0; elapsed=0.0; depth1=0; depth2=0
    while position<len(records):
        record=records[position]; depth=int(chooser(record)); depth=1 if depth<=1 else 2
        keep=1 if record["first_match"] else 0
        if record.get("available_depth",2)==0:
            position+=1; rounds+=1; elapsed+=record["verify_ms"]; depth1+=1
            continue
        if depth==2 and keep==1 and record["second_match"]: keep=2
        accepted+=keep; position+=keep+1; rounds+=1; elapsed+=cost1 if depth==1 else cost2
        depth1+=depth==1; depth2+=depth==2
    generated=min(position+1,len(records)+1)
    return {"generated_tokens":generated,"rounds":rounds,"accepted_drafts":accepted,
            "depth1_rounds":depth1,"depth2_rounds":depth2,"modeled_ms":elapsed,
            "modeled_tps":generated*1000/elapsed}


def nearest_probability(training,margin,field,k=32):
    nearest=sorted(training,key=lambda x:abs(x[0]-margin))[:min(k,len(training))]
    return sum(1 for x in nearest if x[1])/len(nearest) if nearest else 0.0


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--input",required=True);ap.add_argument("--output",required=True)
    args=ap.parse_args(); root=pathlib.Path(args.input)
    data={}
    for fixture in ("code","chat"):
        d1=load(root/f"{fixture}-d1.jsonl",1);d2=load(root/f"{fixture}-d2.jsonl",2)
        if d1["summary"]["token_hash"]!=d2["summary"]["token_hash"]: raise ValueError(f"{fixture}: serial dump hashes differ")
        data[fixture]={"d1":d1,"d2":d2}
    first_train=[(r["first_margin"],r["first_match"]) for x in data.values() for r in x["d2"]["records"] if r["available_depth"]>0]
    second_train=[(r["second_margin"],r["second_match"]) for x in data.values() for r in x["d2"]["records"] if r["first_match"] and r["second_margin"] is not None]
    report={"fixtures":{}}
    thresholds=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0,2.0,4.0]
    for fixture,pair in data.items():
        d1,d2=pair["d1"],pair["d2"];records=d2["records"];cost1=d1["round_ms"];cost2=d2["round_ms"]
        policies={"fixed1":replay(records,cost1,cost2,lambda r:1),
                  "fixed2":replay(records,cost1,cost2,lambda r:2)}
        for threshold in thresholds:
            policies[f"margin_{threshold:g}"]=replay(records,cost1,cost2,lambda r,t=threshold:1 if r["first_margin"]<t else 2)
        def cost_policy(record):
            p1=nearest_probability(first_train,record["first_margin"],"first_match")
            p2=nearest_probability(second_train,record["second_margin"],"second_match") if record["second_margin"] is not None else 0
            return 2 if (1+p1+p1*p2)/cost2>(1+p1)/cost1 else 1
        policies["cost_knn32"]=replay(records,cost1,cost2,cost_policy)
        report["fixtures"][fixture]={"serial_token_hash":d2["summary"]["token_hash"],
             "measured":{"d1_round_ms":cost1,"d2_round_ms":cost2,
                         "d1_verify_ms_median":d1["verify_ms_median"],"d2_verify_ms_median":d2["verify_ms_median"]},
             "policies":policies}
    joint=[(name,min(report["fixtures"][fixture]["policies"][name]["modeled_tps"] for fixture in report["fixtures"]))
           for name in report["fixtures"]["code"]["policies"]]
    report["best_joint"]=max(joint,key=lambda item:item[1])
    pathlib.Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"best_joint":report["best_joint"],"best":{f:max(v["policies"].items(),key=lambda x:x[1]["modeled_tps"]) for f,v in report["fixtures"].items()}},indent=2))


if __name__=="__main__":main()
