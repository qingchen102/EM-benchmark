import argparse
from em_signal_simulator.factory import generate_dataset

def main():
    p=argparse.ArgumentParser(description="Generate complex IQ benchmark samples")
    p.add_argument("--output-dir", default="dataset"); p.add_argument("--count", type=int, default=10)
    p.add_argument("--mod-type", default="QPSK"); p.add_argument("--jamming-type", default="none")
    p.add_argument("--snr-db", type=float, default=10); p.add_argument("--jsr-db", type=float, default=3)
    p.add_argument("--num-samples", type=int, default=1024); p.add_argument("--cfo", type=float, default=0)
    p.add_argument("--mixed", action="store_true", help="随机选择调制、干扰类型和 SNR")
    p.add_argument("--snr-range", nargs=2, type=float, default=(-10, 20), metavar=("MIN", "MAX"))
    a=p.parse_args();
    generate_dataset(a.output_dir, a.count, mixed=a.mixed, snr_range=a.snr_range,
                     mod_type=a.mod_type, jamming_type=a.jamming_type, snr_db=a.snr_db,
                     jamming_to_signal_ratio_db=a.jsr_db, num_samples=a.num_samples, cfo=a.cfo)
if __name__ == "__main__": main()
