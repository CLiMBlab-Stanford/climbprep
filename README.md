# climbprep

An in-progress repository to streamline and standardize preprocessing and analysis
of fMRI data in Stanford's CLiMB Lab.

## Usage

### Cluster job creation/submission

Many of the utilities in this repository are compute/memory intensive and are
thus intended to be run in parallel on the cluster. For convenience, you can
generate batch scripts for different types of jobs and submit them to the
scheduler. To do this, run:

    python -m climbprep.make_jobs <SUBJECT_ID>( <SUBJECT_ID>)* -p <PROJECT_ID> -j <JOB_TYPE>

(the `-p` option can be omitted if the project is `climblab`).
Run the above with `-h` to see all available command line options.

The result will be a collection of files with the suffix `*.pbs` (in the
current working directory by default, although this can be configured),
each containing a batch script for the SLURM scheduler. You can submit
these individually using the standard SLURM interface:

    sbatch <FILE>.pbs

Or you can use the `sbatch.sh` script to submit many at once:

    ./sbatch.sh *.pbs

Logs from stdin/err for each script will be written to a matching file
in the current working directory with the suffix `*.out`.

### Repair

Much of the global repository metadata, especially `participants.tsv`, can
be automatically generated from the project data itself, so a utility is
provided to do this. Currently, repair only updates `participants.tsv`,
and it does so by removing any participants who don't have a subdirectory
and adding any missing participants who do have a subdirectory. It also
automatically generates a column `tasks` containing a comma-delimited
list of all tasks in the participant's `func` folder(s), to enable easy
searching for specific tasks. To perform repair, run:

    python -m climbprep.repair

Because `repair` modifies global files, *it should not be run in parallel*.
Only run it if you are confident that no one else in the group is running
`repair` or editing a `participants.tsv` file.

### BIDSification

Whether working with our own data or data from other labs, the first step is
to convert the data into the BIDS format. We maintain running BIDS "projects"
that are organized around sites rather than particular experiments, to
facilitate secondary analyses. The current set of projects are:

- `climblab` (the main project)
- `evlab` (data from the Fedorenko Lab)

Our general BIDS directory is found at `/juice2/scr2/nlp/climblab/BIDS`.
Each project is a subdirectory. To BIDSify new fMRI source data,
first copy the new source directory (in any format supported by dcm2bids)
to `/juice2/scr2/nlp/climblab/BIDS/<project>/sourcedata/sub-<subjectID>/ses-<sessionID>`,
replacing all bracketed variables with appropriate values. Subject and session
IDs must follow BIDS naming conventions.

It is strongly encouraged (but not technically required) to also create
a CSV table in the source directory above called `runs.csv` that describes
the task details for all functional runs. This table should have at minimum
2 columns: `TaskName` and `SeriesNumber`, where `TaskName` is a name for the
task performed (or `Rest` if no task) and `SeriesNumber` is the integer
number for the run as recorded in the corresponding dicoms. There is a third
(technically optional) field called `EventFile` that provides paths
(relative to the source directory for the subject/session that you are
creating) to the BIDS-style `events.tsv` file containing onsets, durations,
and trial types for all events in the task corresponding to the run.
Please create this unless the run is task-free (e.g., rest).
Creating this table is the main manual labor for experimenters who have
just collected new data. Once the table is created, it greatly facilitates
downstream analyses and also makes the tasks searchable for secondary
analyses.

Once the source directory is created as described above and `runs.csv`
has been added, the data can be BIDSified in one go with the following
command:

    python -m climbprep.bidsify <SUBJECT_ID> -p <PROJECT_ID>

(the `-p` option can be omitted if the project is `climblab`).
Run the above with `-h` to see all available command line options.

If you are BIDSifying the first-ever session from a new subject,
make sure to add their ID to the project's `participants.tsv`
file.

### Preprocessing (fMRIprep)

Preprocessing is handled by fMRIprep, which the `preprocess` script in this
repository simply wraps while also helping to maintain norms in the lab around
default settings and the file organization of the outputs. For this reason,
lab members are encouraged to use this codebase for preprocessing whenever
possible, rather than running fMRIprep directly. To preprocess a participant,
run:

    python -m climblab.preprocess <SUBJECT_ID> -p <PROJECT_ID> <FMRIPREP_ARGS>

(the `-p` option can be omitted if the project is `climblab`).
Run the above with `-h` to see all available command line options.
Any additional command line options will be passed directly to fMRIprep.
This will result in preprocessed derivatives stored at the following
location relative to the BIDS project's root:

    derivatives/fmriprep/<PREPROCESSING_LABEL>/sub-<SUBJECT_ID>

By default, `<PREPROCESSING_LABEL>` is `main`, but this can be configured by
the user. The use of a label for preprocessing is designed to enable multiple
preprocessing configurations for the same data, as required by the analyses.

This utility will also place fMRIprep's working directory in a standard
location that will support resumption if preprocessing gets interrupted.
This is important because preprocessing takes a while (a day or more for
a typical session)!

### First-levels

TODO

