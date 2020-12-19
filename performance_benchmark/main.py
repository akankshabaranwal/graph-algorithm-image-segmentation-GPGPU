import argparse
import os
from tqdm import tqdm
from time import sleep
from threading import Timer
import sys
import subprocess

# Important have optimal params for execution

# Input arguments standard:
# -i input_image
# -o segment_output_image
# -w amount of warmup iterations
# -b amount of benchmark iterations
# -p if you want to time partially
#
# You can pass extra parameters to my python script using --special "-x 1 - y 2 -z 3"
#
# Printing standard:
# - only print relevant CSV information to stdout, print everything else to stderr
# - don't print warmup round timings to stdout
# - Use the headers gaussian, graph, segment, output, complete to indicate what a particular CSV column corresponds to

# TODO: write to one CSV
# TODO: lots of small segments in naive algo makes hard to compare
# TODO: add benchmarking flag to code to avoid writing to disk


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


def getImagePaths(directory):
    image_extensions = [".jpg", ".png", ".gif", ".ppm"]
    image_paths = []
    for file in os.listdir(directory):
        if file.endswith(tuple(image_extensions)):
            image_paths.append(os.path.join(directory, file))

    return image_paths


def createOutImagePath(image_path, segoutdir):
    pre, ext = os.path.splitext(os.path.basename(image_path))
    outimage = pre + "_segmented" + ".png"

    return os.path.join(segoutdir, outimage)


def createOutCSVPath(image_path, benchoutdir):
    pre, ext = os.path.splitext(os.path.basename(image_path))
    outimage = pre + ".csv"

    return os.path.join(benchoutdir, outimage)


def createErrorPath(image_path, benchoutdir):
    pre, ext = os.path.splitext(os.path.basename(image_path))
    outimage = pre + "_errorlog" + ".txt"

    return os.path.join(benchoutdir, outimage)


def createInfoPath(benchoutdir):
    infofile = "bench_info.txt"

    return os.path.join(benchoutdir, infofile)


def createParams(executable, image, segment_output_directory, n_warmup, n_benchmark, partial, special_args):
    output_path = createOutImagePath(image, segment_output_directory)

    command_params_print = [
        executable,
        "-i", image,
        "-o", output_path,
        "-w", str(n_warmup),
        "-b", str(n_benchmark),
    ]

    command_params = [
        os.path.abspath(executable),
        "-i", os.path.abspath(image),
        "-o", os.path.abspath(output_path),
        "-w", str(n_warmup),
        "-b", str(n_benchmark),
    ]

    if partial:
        command_params.append("-p")

    if special_args:
        command_params.extend(special_args.split(" "))

    return command_params, command_params_print


def execute(executable, image, segment_output_directory, bench_output_directory, n_warmup, n_benchmark, partial, special_args, timeout):
    command_params, command_params_print = createParams(executable, image, segment_output_directory, n_warmup, n_benchmark, partial, special_args)

    command_msg = " ".join([str(el) for el in command_params])
    tqdm.write("Executing {} {}".format(executable, command_msg))

    # Execute program
    p1 = subprocess.Popen(command_params, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        stdout, stderr = p1.communicate(timeout=timeout)
        exitcode = p1.returncode
        if exitcode != 0:
            # Write error log
            error = stderr.decode('ascii')
            tqdm.write("Finished with exit code {}.".format(exitcode))
            error_out = open(createErrorPath(image, bench_output_directory), "w")
            error_out.write(error)
            error_out.close()
            return exitcode
        else:
            # Write output CSV
            output = stdout.decode('ascii')
            tqdm.write("Finished correctly with exit code 0".format(exitcode))
            csv_out = open(createOutCSVPath(image, bench_output_directory), "w")
            csv_out.write(output)
            csv_out.close()
            return exitcode
    except subprocess.TimeoutExpired as e:
        p1.kill()
        tqdm.write("Execution timed out (max time {}s)".format(timeout))
        return -1


def execute_wrapper(executable, image, segment_output_directory, bench_output_directory, n_warmup, n_benchmark, partial, special_args, retries, timeout):
    for i in range(0, retries+1):
        exitcode = execute(executable, image, segment_output_directory, bench_output_directory, n_warmup, n_benchmark, partial, special_args, timeout)
        if (exitcode == 0):
            break
        else: # Failed
            tqdm.write("Retrying...")


def createInfoFile(input_directory, segment_output_directory, bench_output_directory, executable, n_warmup, n_benchmark, partial, special_args, retries, timeout):
    info = "* Executable: {}".format(executable)
    info += "* Input directory: {}".format(input_directory)
    info += "* Segments directory: {}".format(segment_output_directory)
    info += "* Benchmark output directory: {}".format(bench_output_directory)
    info += "* Special arguments used: {}\n".format(special_args) if special_args is not None else "* No special arguments used\n"
    info += "* Warmup iterations: {}".format(n_warmup)
    info += "* Benchmark iterations: {}".format(n_benchmark)
    info += "* Type: {}".format("partial" if partial else "complete")
    info += "* Retries: {}".format(retries)
    info += "* Timeout: {}".format(timeout)

    info_out = open(createInfoPath(bench_output_directory), "w")
    info_out.write(info)
    info_out.close()


def benchmark(input_directory, segment_output_directory, bench_output_directory, executable, n_warmup, n_benchmark, partial, special_args, retries, timeout):
    print("Starting {} benchmark with {} warmup iterations and {} benchmark iterations...".format("partial" if partial else "complete", n_warmup, n_benchmark))
    print("* Executable: {}".format(executable))
    print("* Input directory: {}".format(input_directory))
    print("* Segments directory: {}".format(segment_output_directory))
    print("* Benchmark output directory: {}".format(bench_output_directory))
    print("* Special arguments used: {}\n".format(special_args) if special_args is not None else "* No special arguments used\n")

    # Get paths to all images (not recursive!)
    image_paths = getImagePaths(input_directory)
    print("Found {} images in directory {}\n".format(len(image_paths), input_directory))

    print("Starting execution ...")

    # Write info file with running info
    createInfoFile(input_directory, segment_output_directory, bench_output_directory, executable, n_warmup, n_benchmark, partial, special_args, retries, timeout)

    for i in tqdm(range(0, len(image_paths)), total=len(image_paths), desc="Benchmarking progress", unit= " images", file=sys.stdout):
        execute_wrapper(executable, image_paths[i], segment_output_directory, bench_output_directory, n_warmup, n_benchmark, partial, special_args, retries, timeout)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Input directory (eg ./bds500/)', type=dir_path, required=True)
    parser.add_argument('-s', help='Segment output directory (eg ./bds500_segments)', type=dir_path, required=True)
    parser.add_argument('-o', help='Output directory (eg ./bds500_measurements)', type=dir_path, required=True)
    parser.add_argument('-e', help='Executable path (eg ../src/build/felz)', type=file_path, required=True)
    parser.add_argument('-w', help='Number of warmup iterations (eg 3)', type=int, required=True)
    parser.add_argument('-b', help='Number of benchmark iterations (eg 20)', type=int, required=True)
    parser.add_argument('-p', action='store_true', help='If you want to time different parts of the execution')
    parser.add_argument('-r', help='Amount of retries when execution fails', type=int, required=True)
    parser.add_argument('-t', help='Timeout after which to stop execution', type=int, required=True)
    parser.add_argument('--special', type=str, help='special arguments to add for executing (eg "-x 1 -y 2 -z")')
    arg = parser.parse_args()

    benchmark(arg.i, arg.s, arg.o, arg.e, arg.w, arg.b, arg.p, arg.special, arg.r, arg.t)


