"""Command line entry points; generation and public-only analysis are distinct."""
import argparse
import json
from pathlib import Path
from .config import Scenario
from .runner import run_scenario, estimate_resources
from .analysis import analyze_observations, load_public
from .export import write_json, legacy_manifest
from .replay import replay_scenario


def main(argv=None):
    parser=argparse.ArgumentParser(description="Static urban multi-station IQ simulator")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("run","validate"):
        p=sub.add_parser(name)
        p.add_argument("--config",type=Path,required=True)
        if name=="run":
            p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("analyze")
    p.add_argument("--public",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("legacy-manifests")
    p.add_argument("--public",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p=sub.add_parser("replay")
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.command in ("run","validate"):
            s=Scenario.model_validate_json(args.config.read_text(encoding="utf-8"))
            result=run_scenario(s,args.output) if args.command=="run" else estimate_resources(s)
        elif args.command=="replay":
            result=replay_scenario(args.run,args.output)
        elif args.command=="analyze":
            result=analyze_observations(args.public)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.output,result)
        else:
            public=load_public(args.public)
            args.output.mkdir(parents=True,exist_ok=True)
            for r in public["receivers"]:
                tasks=[t for t in public["tasks"] if r["receiver_id"] in t["receiver_ids"]] or [None]
                for task in tasks:
                    suffix=task["id"] if task else "generic"
                    target=(args.output/f'{r["receiver_id"]}_{suffix}.json').resolve()
                    if not target.is_relative_to(args.output.resolve()):
                        raise ValueError("manifest path escapes output directory")
                    write_json(target,
                               legacy_manifest(args.public,r,task))
            result={"status":"exported","output":str(args.output)}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,RuntimeError,FileExistsError,OSError) as e:
        parser.exit(2,f"cityrf: {e}\n")
