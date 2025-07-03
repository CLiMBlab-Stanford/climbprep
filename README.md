# climbprep

An in-house repository to streamline and standardize preprocessing and analysis
of fMRI data in Stanford's CLiMB Lab. This codebase based is designed
to be run on the lab's cluster and will only work there, because it uses/enforces
assumptions about the organization of our central data store. However,
external users are welcome to fork the codebase and adjust the underlying paths
(`climbprep/constants.py`) to suit their own needs.


## Getting Started

New lab members can get everything initialized in one go by running:

    python -m climbprep.quickstart

Note that this script is interactive and your responses will be required at points.
QUICKSTART NOT YET IMPLEMENTED!

Alternatively, you can set manually. This will involve setting up your `~/.profile` and 
`~/.bashrc` files following the instructions in the 
[cluster onboarding docs](https://docs.google.com/document/d/1nlpqFCRX4wo-gqa84rA9X0DaYYIPNFjB9dgFl3RUjlQ/edit),
adding the environment variables from the `BASHRC` constant in 
`climbprep/constants.py`, installing [Anaconda](https://www.anaconda.com/) to
`/nlp/scr/USERNAME/miniconda3/bin/conda`, and installing the `climbprep` environment
like so:

```bash
conda env create -f climbprep/resources/conda.yml
```

This will create a new environment called `climbprep` with all the necessary dependencies,
which you can activate by running:

```bash
conda activate climbprep
```


## Preliminaries

Our general BIDS super-directory is found at `/juice6/u/nlp/climblab/BIDS`.
Each subdirectory is a BIDS project. New projects can be created as needed.
The lab's main project is `climblab`, which contains all the non-development
data that we have collected on site. Development data (data collected on
development hours at the CNI) cannot be used in published research and must
be put in the `dev` project.

The `climblab` project has a somewhat unusual organization: although we
often collect multiple sessions from an individual, the project does
not have a multisession structure. Instead, the `participant` level of
the project indexes the session based on its `flywheel` ID, irrespective
of whether other data from that participant exists in the project.
There are two reasons for this:
1. The mapping from sessions to projects is often many-to-many
2. We want to avoid multisession aggregation by fMRIprep, which can change published results

Instead, the identity of the participant is tracked via the `climblab_id`
field of `participants.tsv`, which is a unique identifier for each
individual. This lets you easily find all sessions from a participant.
The `climblab_id` can only be maintained by core lab members with access
to the high-risk database, since we need identifiers to link
sessions to participants. You can contact the core team as needed if
you are unable to find the ID for a participant (e.g., for newly-collected
data).

If you want to do multisession analyses, the correct approach is to
"rescaffold" the relevant subset of `climblab` data into a new BIDS
project. This codebase provides a utility for doing so
(see `rescaffold` below).


## Usage: Core Functions

Full help strings for all the utilities below can be obtained by running
the command with the `-h` argument. Note that when you are on the cluster,
as long as you have run `quickstart`, `climbprep` will be in your `PATH`
and available anywhere, and you can actually run all the commands below
without the `python -m climbprep.` (i.e., `python -m climbprep.preprocess`
will be equivalent to `preprocess`).

### bidsify

Whether working with our own data or data from other labs, the first step is
to convert the data into the BIDS format. We maintain running BIDS "projects"
that are organized around sites rather than particular experiments, to
facilitate secondary analyses. The current set of projects are:

- `climblab` (the main project)
- `evlab` (data from the Fedorenko Lab)

To BIDSify new fMRI source data,
first copy the new source directory (in any format supported by `dcm2bids`)
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
To view the available config options, see `CONFIG['preprocess']` in 
`climbprep/constants.py`.

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
see `CONFIG['clean']` in `climbprep/constants.py`.

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
statmaps) from task fMRI data. Modeling uses the 
[fitlins](https://fitlins.readthedocs.io/en/latest/index.html) library.
It can only be run after preprocessing is complete. To model a participant,
run:

    python -m climblab.model <PARTICIPANT_ID> -p <PROJECT_ID>

(the `-p` option can be omitted if the project is `climblab`).
The `-c` argument can be omitted (defaulting to `T1w` standard settings),
or set to either `mni` or `T1w` to respectively use standard
settings for MNI space or native space, or set to a path containing
a YAML configuration file. To view the available config options,
see `CONFIG['model']` in `climbprep/constants.py`.

The lab maintains a library of model files conforming to
the [BIDS Stats Models specification]
(https://bids-standard.github.io/stats-models) at 
`/juice6/u/nlp/climblab/modelfiles`.
By default, `model` will use the set of tasks available from the participant
to infer which models in the library are relevant and run them all, with
outputs saved to:

    derivatives/firstlevels/<MODEL_LABEL>/<MODEL_NAME>/sub-<PARTICIPANT_ID>

where `MODEL_LABEL` refers to the configuration name (e.g., `mni`)
and `MODEL_NAME` refers to the standard  name for the model (`PREFIX` 
in `PREFIX_model.json` in the model's filename). To run
a subset of available models, you can pass their names as a space-delimited
list using the `-m` option. If you need to make a new modelfile (e.g.,
you designed and ran a new experiment), you can either write your own
BIDS-conformant modelfile by hand, or use the `generate_model`
utility (see below).

How do you decide whether a model variant is a reparameterization of the
same model (a different `MODEL_LABEL`) or a different model (a different
`MODEL_NAME`)? The distinction is a bit arbitrarily determined by the
BIDS Stats Models specification: any variation that requires a different
`model.json` file is a different model (`MODEL_NAME`), whereas any variation
that is just a different function call of `fitlins` on the same model file
is a reparameterization (`MODEL_LABEL`). For example, the `mni` and `T1w`
configurations are both reparameterizations of the same model, because
they both use the same model file, but they differ in the space in which
the model is fit (MNI vs. native space). In contrast, different sets of
nuisance regressors (movement, physiological noise, etc.) must be
specified in the `model.json` file, so they are different models
(`MODEL_NAME`), even if they are fit in the same space (e.g., MNI)
using the same task data and conditions. If you're not sure how to
implement a variant, consult the docs for `fitlins` and `BIDS Stats Models`.


### plot
The plotting step generates surface plots of the results of the modeling
step. It can only be run after modeling is complete. To plot a participant's
data, run:

    python -m climblab.plot <PARTICIPANT_ID> -p <PROJECT_ID>

(the `-p` option can be omitted if the project is `climblab`).
The `-c` argument can optionally be used to specify plotting configuration
parameters, including which model type to plot (defaults to `T1w`).
To view the available config options, see `CONFIG['plot']` in
`climbprep/constants.py`.


## Usage: Helper Functions


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


### generate_model

The lab uses the BIDS Stats Models specification to control first-level models,
where each model written as a JSON file in `/juice6/u/nlp/climblab/modelfiles`.
For new experiments or analyses, you may need to make a new model file.
You're welcome to do this by hand following the BIDS specification, but
this can be pretty error-prone. For typical experiments, most of the 
model-writing can be automated by the `generate_model` utility, which will
generate a model file based on a set of task names and conditions inferred
from an example participant, along with any novel contrasts you want, which you
specify in a YAML file.

To generate a new model, run:

    python -m climbprep.generate_model <PARTICIPANT_ID> <TASK>(<TASK>)* -c <CONFIG>

Where `<PARTICIPANT_ID>` is the ID of a BIDSified participant whose events files you
want to use to populate the model's conditions. The `-c` argument is the path
to a YAML file with the following structure:
```
LEVEL:
  CONTRAST:
    CONDITION1: WEIGHT
    CONDITION2: WEIGHT
    ...
```
where `LEVEL` is the level of the model (one of `run`, `session`, or `subject`).
For example, to specify a sentences vs. nonwords contrast at the run level,
you would write:
```
run:
  SvN:
    S: 0.5
    N: -0.5
```
Some contrasts may compare conditions that do not appear in all runs,
in which case it would not be possible to compute them at the run level
and you need to specify them at the session and/or subject level.
Somewhat counterintuitively, the session and subject levels are computed
independently from the run level, so new contrasts at the session level
do not propagate to the subject level. If you want the contrast to exist
at both levels, you must specify it at both levels.

Note that the weight magnitudes should sum to one, to facilitate comparison
across contrasts.



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


### rescaffold

If you want to run multisession analyses, you need to "rescaffold" the
relevant subset of `climblab` data into a new BIDS project so that fMRIprep
can correctly transform the multisession data into a shared anatomical space.
You can rescaffold the entire dataset, which will produce a copy of `climblab`
in which sessions are organized by subject. You can also provide a set
of `climblab_id`'s or task names (either space-delimited in the command line
or a path to a textfile containing the list, one item per line) to rescaffold 
only a subset of the data. To rescaffold, run:

    python -m climbprep.rescaffold <OUTPUT_PROJECT_ID> [<CLIMBLAB_ID> ...]

This tool will provide minimal BIDS metadata to the new project to pass
validation, but it is strongly recommended to enrich the metadata before
publication/release.


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


