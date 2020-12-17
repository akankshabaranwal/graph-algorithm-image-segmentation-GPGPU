"""benchmark_segments.py: Correctness benchmark on the BSDS500 dataset.

Usage: scripts/benchmark_segments.py CSV_FILE DUMP_PATH COMMAND

Options:
  -h --help                   Show this screen.

Command:
  This benchmarking script calls the felzenswalb implementation with the input
  file and output file substituted using the `@INPUT@' and `@OUTPUT@' keywords 
  in your COMMAND description. The output images are saved in the DUMP_PATH
  and the Achievable Segmentation Accuracy (ASA) and the Under-segmentation
  Error (UE) scores are saved in the CSV_FILE.
  
  For example, with the cuda-mst-naive implementation the COMMAND would be:
  "./out/felz -w 0 -b 1 -k 20 -E 1.0 -i '@INPUT@' -o '@OUTPUT@'"

Example:
  python3 scripts/benchmark_segments.py \\
      csv/cuda-mst-naive-K20-E1.csv     \\
      dumps/cuda-mst-naive-K20-E1/      \\
      "./out/felz -w 0 -b 1 -k 20 -E 1.0 -i '@INPUT@' -o '@OUTPUT@'"
"""

import csv
import datetime
import os
import pathlib
import sys
import time

import numpy
from docopt import docopt

from compare_segments import asa_score, find_segments, underseg_error

if __name__ == "__main__":
    args = docopt(__doc__)

    input_dir = pathlib.Path('dataset/BSDS500/images/')
    ground_truth_dir = pathlib.Path('dataset/BSDS500/ground_truth/')
    if not input_dir.exists() or not ground_truth_dir.exists():
        print("Cannot find BSDS500 in 'dataset/', please run from the root of the project as 'scripts/benchmark_segments.py'")
        sys.exit(1)

    command = args['COMMAND']
    if not '@INPUT@' in command or not '@OUTPUT@' in command:
        print("Specified COMMAND does not contain `@INPUT@' and/or `@OUTPUT@' keywords. Exiting.")
        sys.exit(1)

    executable = pathlib.Path(command.split()[0])
    if not executable.exists():
        print(f"Specified executable in COMMAND, '{executable}', does not exist. Exiting.")
        sys.exit(1)

    dump_dir = pathlib.Path(args['DUMP_PATH'])
    dump_dir.mkdir(parents=True, exist_ok=True)
    if any(dump_dir.iterdir()):
        print(f"Specified DUMP_PATH, '{dump_dir}', is not empty. Exiting.")
        sys.exit(1)

    try:
        csv_file = open(args['CSV_FILE'], 'w')
        csv_writer = csv.writer(csv_file)
    except:
        print(f"Failed to create CSV_FILE, '{args['CSV_FILE']}'. Exiting.")
        sys.exit(1)

    csv_writer.writerow(['output', 'ground_truth', 'asa_score', 'ue_score'])
    
    print(f"Started at {datetime.datetime.utcnow() + datetime.timedelta(+1, 3600)} UTC+1 (Europe/Zurich)")
    print()

    start = time.time()

    for idx, input_path in enumerate(sorted(input_dir.glob('*.jpg'))):
        output_path = dump_dir / (input_path.stem + '_OUT.png')

        new_command = command.replace('@INPUT@', str(input_path))
        new_command = new_command.replace('@OUTPUT@', str(output_path))

        print(f"** {idx + 1:3} / 500 COMMAND = {new_command}")
        
        os.system(new_command)
        while not output_path.exists():
            print("**     /     COMMAND ^ Failed to generate output. Re-trying in 1 second.")
            time.sleep(1)
            os.system(new_command)
        

        for ground_truth_path in sorted(ground_truth_dir.glob(input_path.stem + '*')):
            print(f"**     /     Comparing with {ground_truth_path}")

            with open(output_path, 'rb') as f:
                segments_in = find_segments(f)

            with open(ground_truth_path, 'rb') as f:
                segments_gt = find_segments(f)
            
            in_N, gt_N = numpy.unique(segments_in).size, numpy.unique(segments_gt).size 
            print(f"**     /     Input segments: {in_N} Ground truth segments: {gt_N}")

            asa, useg = asa_score(segments_in, segments_gt), underseg_error(segments_in, segments_gt)
            print(f"**     /     ASA: {asa:.3f} USErr: {useg:.3f}")

            csv_writer.writerow([output_path, ground_truth_path, asa, useg])
        
        since = time.time() - start
        eta = (since / (idx + 1)) * (500 - idx)

        print(f"**     /     Finished. Time: {since:.2f}s ETA: {eta:.2f}s")

    print()
    print(f"Ended at   {datetime.datetime.utcnow() + datetime.timedelta(+1, 3600)} UTC+1 (Europe/Zurich)")
    csv_file.close()
