# climbprep

An in-house repository to streamline and standardize preprocessing and analysis
of fMRI data in Stanford's CLiMB Lab.


## Installation

Installation is just a matter of setting up an [Anaconda](https://www.anaconda.com/) environment
with the right software dependencies. Once you have Anaconda installed, you can create the
environment by running the following command in the terminal:

```bash
conda env create -f conda.yml
```

This will create a new environment called `climbprep` with all the necessary dependencies,
which you can activate by running:

```bash
conda activate climbprep
```


## Modifying the pipeline

Contributions to this codebase from lab members are welcome! Just please make sure
to work in a branch/fork and submit a pull request for review before merging
any changes to the main branch.

One probably frequent modification scenario is if there's some new configuration
for one of these steps that is likely to be generally useful to the lab, and you
want to make it conveniently available lab-wide by keyword rather than having
to pass around config files. The codebase is set up to facilitate these kinds of
changes by placing all constants, including default configurations, in the
`climbprep/constants.py` file, allowing them to be easily changed or expanded
without affecting core functionality. If you want to expand existing keyword-
accessible configurations, you can freely add configurations to `constants.py`.


## Usage

Full help strings for all the utilities below can be obtained by running
the command with the `-h` argument.


### make_jobs

Many of the utilities in this repository are compute/memory intensive and are
thus intended to be run in parallel on the cluster. For convenience, you can
generate batch scripts for different types of jobs and submit them to the
scheduler. To do this, run:

    python -m climbprep.make_jobs <PARTICIPANT_ID>( <PARTICIPANT_ID>)* -p <PROJECT_ID>

(the `-p` option can be omitted if the project is `climblab`).
By default, this will run the participant (and if relevant any component sessions)
sequentially through the entire climbprep pipeline, from bidsification through cleaning.
If you have a subset of steps you want to run, you can specify them as a space-delimited
list using the `-j` option.

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


### repair

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


### bidsify

Whether working with our own data or data from other labs, the first step is
to convert the data into the BIDS format. We maintain running BIDS "projects"
that are organized around sites rather than particular experiments, to
facilitate secondary analyses. The current set of projects are:

- `climblab` (the main project)
- `evlab` (data from the Fedorenko Lab)

Our general BIDS directory is found at `/juice6/u/nlp/climblab/BIDS`.
Each project is a subdirectory. To BIDSify new fMRI source data,
first copy the new source directory (in any format supported by dcm2bids)
to `/juice6/u/nlp/climblab/BIDS/<project>/sourcedata/sub-<subjectID>`,
replacing all bracketed variables with appropriate values. Subject
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

    python -m climbprep.bidsify <PARTICIPANT_ID> -p <PROJECT_ID> -c <CONFIG>

(the `-p` option can be omitted if the project is `climblab`).
The optional `-c` argument can be used to specify a `dcm2bids`-compatible
configuration file, although there should rarely be a need to deviate
from the default (automatically-generated) config.

If you are BIDSifying the first-ever session from a new subject,
make sure to add their ID to the project's `participants.tsv`
file.


### preprocess

Preprocessing is handled by fMRIprep, which the `preprocess` script in this
repository simply wraps while also helping to maintain norms in the lab around
default settings and the file organization of the outputs. For this reason,
lab members are encouraged to use this codebase for preprocessing whenever
possible, rather than running fMRIprep directly. To preprocess a participant,
run:

    python -m climblab.preprocess <PARTICIPANT_ID> -p <PROJECT_ID> -c <CONFIG>

(the `-p` option can be omitted if the project is `climblab`).
The optional `-c` argument can be used to specify a YAML configuration
containing fMRIprep CLI args, which will be passed directly to fMRIprep.
This will result in preprocessed derivatives stored at the following
location relative to the BIDS project's root:

    derivatives/fmriprep/<PREPROCESSING_LABEL>/sub-<PARTICIPANT_ID>

By default, `<PREPROCESSING_LABEL>` is `main`, but this can be changed in the
config. The use of a label for preprocessing is designed to enable multiple
preprocessing configurations for the same data, as required by different analyses.

This utility will also place fMRIprep's working directory in a standard
location that will support resumption if preprocessing gets interrupted.
This is important because preprocessing takes a while (a day or more for
a typical session)!


### clean

The cleaning step simplifies and standardizes the lab's procedures for
standardizing/denoising preprocessed data, typically in preparation for
subsequent functional connectivity analysis. The purpose of cleaning is
to remove unwanted confounds from the data and (optionally) censoring
high-motion volumes. The cleaning step can only be run after
preprocessing is complete. To clean a participant's data,
run:

    python -m climblab.clean <PARTICIPANT_ID> -p <PROJECT_ID> -c <CONFIG>

(the `-p` option can be omitted if the project is `climblab`).
The `-c` argument can be omitted (defaulting to FC standard settings),
or set to either `fc` or `firstlevels_like` to respectively use default
settings for functional connectivity (rescaling and scrubbing) or 
firstlevels estimation (no rescaling or scrubbing), or set to a path
containing a YAML configuration file. To view the available config options,
see `DEFAULTS['clean']['firstlevels']` in `climbprep/constants.py`.

Cleaning will result in cleaned derivatives stored at the following
location relative to the BIDS project's root:

    derivatives/cleaned/<CLEANING_LABEL>/sub-<PARTICIPANT_ID>

In addition to cleaned images, this directory will contain a `samplemask.tsv`
file for each cleaned run, which contains the indices of *retained* volumes
under scrubbing (i.e., removal) of volumes with excessive head motion. These
files will be produced regardless of whether scrubbing was applied during
cleaning (this is a configurable setting), because some `nilearn` functions
allow these sample masks to be passed as a parameter to functions.


### model

The modeling step generates first-level analyses (participant-specific
statmaps) from task fMRI data. It can only be run after preprocessing
is complete. o clean a participant's data,
run:

    python -m climblab.model <PARTICIPANT_ID> -p <PROJECT_ID>

(the `-p` option can be omitted if the project is `climblab`).
The `-c` argument can be omitted (defaulting to MNI standard settings),
or set to either `mni` or `T1w` to respectively use standard
settings for MNI space or native space, or set to a path containing
a YAML configuration file. To view the available config options,
see `DEFAULTS['model']['main']` in `climbprep/constants.py`.

The lab maintains a library of model files conforming to
the [BIDS Stats Models specification]
(https://bids-standard.github.io/stats-models) at `/juice6/u/nlp/climblab/`.
By default, `model` will use the set of tasks available from the participant
to infer which models in the library are relevant and run them all, with
outputs saved to:

    derivatives/firstlevels/<MODEL_LABEL>/<MODEL_NAME>/sub-<PARTICIPANT_ID>

where `MODEL_LABEL` refers to the configuration name (e.g., `mni`)
and `MODEL_NAME` refers to the standard  name for the model (`PREFIX` 
in `PREFIX_model.json` in the model's filename). To run
a subset of available models, you can pass their names as a space-delimited
list using the `-m` option.
