import argparse
import json
from .runner import run_baseline

p=argparse.ArgumentParser(description="Classical public-IQ detection and bearing localization")
p.add_argument("--public",required=True)
p.add_argument("--output",required=True)
a=p.parse_args()
print(json.dumps(run_baseline(a.public,a.output),indent=2))
