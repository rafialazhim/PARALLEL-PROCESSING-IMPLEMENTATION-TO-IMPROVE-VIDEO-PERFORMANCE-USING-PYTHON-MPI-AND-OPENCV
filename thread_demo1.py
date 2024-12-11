from mpi4py import MPI
import argparse
import os
import cv2
import datetime
from CountsPerSec import CountsPerSec

def putIterationsPerSec(frame, iterations_per_sec):
    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
                (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    return frame

def logIterations(iterations_per_sec):
    """Log the iterations per second to a file."""
    log_dir = "C:/Logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, "video_log.txt")
    with open(log_file_path, "a") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - Iterations per second: {iterations_per_sec}\n")

def noThreading(source=0, rank=0):
    """Grab and show video frames without multithreading."""

    cap = cv2.VideoCapture(source)
    cps = CountsPerSec().start()

    while True:
        grabbed, frame = cap.read()
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break

        # Compute iterations per second
        iterations_per_sec = cps.countsPerSec()
        
        # Display the iterations per second on the frame
        frame = putIterationsPerSec(frame, iterations_per_sec)
        
        # If rank 0, display the video frame
        if rank == 0:
            cv2.imshow("Video", frame)
        
        # If rank 1, log the iterations per second
        if rank == 1:
            logIterations(iterations_per_sec)
        
        # Increment counts per second
        cps.increment()

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default="C:/video/video1.mp4",
                    help="C:/video/video2.mp4 (default 0).")
    ap.add_argument("--thread", "-t", default="none",
                    help="Threading mode: get (video read in its own thread),"
                         + " show (video show in its own thread), both"
                         + " (video read and video show in their own threads),"
                         + " none (default--no multithreading)")
    args = vars(ap.parse_args())

    # Check if source is a valid file or camera ID
    if (
        isinstance(args["source"], str)
        and args["source"].isdigit()
        and not os.path.isfile(args["source"])
    ):
        args["source"] = int(args["source"])

    # Run video processing based on MPI rank
    if rank in [0, 1]:  # Process 0 will display video, Process 1 will log
        noThreading(args["source"], rank)

    # Synchronize processes if needed
    comm.Barrier()

if __name__ == "__main__":
    main()
