import argparse
import os
from glob import glob
import sys
import pandas as pd
import math

from src.logger import LOGGER

from src.validate import ( 
    valid_output_dir,
    valid_readable_dir,
    valid_subj_ses_ID,
    valid_whole_number,
    valid_readable_file
)

from src.utilities import (
    dict_has,
    ensure_prefixed,
    get_age_closest_to,
    only_Ts_needed_for_bibsnet_model
)

SCRIPT_DIR_ARG = "--script-dir"
SCRIPT_DIR = os.path.dirname(os.path.dirname(__file__))
AGE_TO_HEAD_RADIUS_TABLE = os.path.join(SCRIPT_DIR, "data",
                                        "age_to_avg_head_radius_BCP.csv")


def get_params(stage_names):
    """
    :param stage_names: List of strings; each names a stage to run
    :return: Dictionary containing all parameters
    """
    default_end_stage = stage_names[-1]
    default_fsl_bin_path = "/opt/fsl-6.0.5.1/bin/"
    default_nnUNet_configuration = "3d_fullres"
    default_nnUNet_predict_path = "/opt/conda/bin/nnUNet_predict"

    msg_stage = ("Name of the stage to run {}. By default, this will be "
                 "the {} stage. Valid choices: {}")
    parser = argparse.ArgumentParser("CABINET")

    # BIDS-App required positional args, validated later in j_args
    parser.add_argument(
        "bids_dir", type=valid_readable_dir,
        help=("Valid absolute path to existing base study directory "
              "containing BIDS-valid input subject data directories. "
              "Example: /path/to/bids/input/")  # TODO Keep as j_args[common][bids_dir]
    )
    parser.add_argument(
        "output_dir", type=valid_readable_dir,  # TODO Does this dir have to already exist?
        help=("Valid absolute path to existing derivatives directory to save "
              "each stage's outputs by subject session into. Example: "
              "/path/to/output/derivatives/")
    )
    parser.add_argument(
        "analysis_level", choices=["participant"],  # TODO Will we ever need to add group-level analysis functionality? Currently this argument does absolutely nothing
        help=("Processing level. Currently the only choice is 'participant'. "
              "See BIDS-Apps specification.")
    )

    # Required flag arguments
    parser.add_argument(
        "-participant", "--subject", "-sub", "--participant-label",
        dest="participant_label", type=valid_subj_ses_ID, required=True,
        help=("The participant's unique subject identifier, without 'sub-' "
              "prefix. Example: 'ABC12345'")  # TODO Make CABINET able to accept with OR without 'sub-' prefix
    )

    # Optional flag arguments
    parser.add_argument(
        "-age", "-months", "--age-months", type=valid_whole_number,
        help=("Positive integer, the participant's age in months. For "
              "example, -age 5 would mean the participant is 5 months old. "
              "Include this argument unless the age in months is specified in "
              "each subject's sub-{}_sessions.tsv file inside its BIDS input directory "
              "or inside the participants.tsv file inside the BIDS directory at the" 
              "subject-level.")
    )
    parser.add_argument(
        "-end", "--ending-stage", dest="end",
        choices=stage_names, default=default_end_stage,
        help=msg_stage.format("last", default_end_stage, ", ".join(stage_names))
    )
    parser.add_argument(
        "--fsl-bin-path",
        type=valid_readable_dir,
        default=default_fsl_bin_path,
        help=("Valid path to fsl bin."
              "Defaults to the path used by the container: {}".format(default_fsl_bin_path))
    )
    parser.add_argument(
        "-model", "--model-number", "--bibsnet-model",
        type=valid_whole_number, dest="model",
        help=("Model/task number for BIBSnet. By default, this will be "
              "inferred from {} based on which data exists in the "
              "--bids-dir. BIBSnet will run model 514 by default for T1w-"
              "only, model 515 for T2w-only, and model 552 for both T1w and "
              "T2w.".format(os.path.join(SCRIPT_DIR, "data", "models.csv")))
    )
    parser.add_argument(
        "--nnUNet", "-n", type=valid_readable_file, default=default_nnUNet_predict_path,
        help=("Valid path to existing executable file to run nnU-Net_predict. "
              "By default, this script will assume that nnU-Net_predict will "
              "be the path used by the container: {}".format(default_nnUNet_predict_path))
    )
    parser.add_argument(
        "--nnUNet-configuration", dest="nnUNet_configuration",
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        default=default_nnUNet_configuration,
        help=("The nnUNet configuration to use."
              "Defaults to {}".format(default_nnUNet_configuration))
    )
    parser.add_argument(
        "--overwrite", "--overwrite-old",  # TODO Change this to "-skip"
        dest="overwrite", action="store_true",
        help=("Include this flag to overwrite any previous CABINET outputs "
              "in the derivatives sub-directories. Otherwise, by default "
              "CABINET will skip creating any CABINET output files that "
              "already exist in the sub-directories of derivatives.")
    )
    parser.add_argument(
        "-ses", "--session", "--session-id", type=valid_subj_ses_ID,
        help=("The name of the session to processes participant data for. "
              "Example: baseline_year1")
    )
    parser.add_argument(
        "-start", "--starting-stage", dest="start",
        choices=stage_names, default=stage_names[0],   # TODO Change default to start where we left off by checking which stages' prerequisites and outputs already exist
        help=msg_stage.format("first", stage_names[0], ", ".join(stage_names))
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help=("Include this flag to print detailed information and every "
              "command being run by CABINET to stdout. Otherwise CABINET "
              "will only print warnings, errors, and minimal output.")
    )
    parser.add_argument(
        "-w", "--work-dir", type=valid_output_dir, dest="work_dir",
        default=os.path.join("/", "tmp", "cabinet"),
        help=("Valid absolute path where intermediate results should be stored."
              "Example: /path/to/working/directory")
    )
    parser.add_argument(
        "-z", "--brain-z-size", action="store_true",
        help=("Include this flag to infer participants' brain height (z) "
              "using the sub-{}_sessions.tsv or participant.tsv brain_z_size column." 
              "Otherwise, CABINET will estimate the brain height from the participant "
              "age and averages of a large sample of infant brain heights.")  # TODO rephrase
    )
    parser.add_argument(
        SCRIPT_DIR_ARG, dest=as_cli_attr(SCRIPT_DIR_ARG),
        type=valid_readable_dir,
        help=("Valid path to the existing parent directory of this run.py "
              "script. Include this argument if and only if you are running "
              "the script as a SLURM/SBATCH job.")
    )
    return validate_cli_args(vars(parser.parse_args()), stage_names,
                             parser)


def validate_cli_args(cli_args, stage_names, parser):
    """
    :param cli_args: Dictionary containing all command-line input arguments
    :param stage_names: List of strings naming stages to run (in order)
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :return: Tuple of 2 objects:
        1. Dictionary of validated parameters
        2. List of dicts which each map "subject" to the subject ID string,
           "age_months" to the age in months (int) during the session, & maybe
           also "session" to the session ID string. Each will be j_args[IDs]
    """
    # Get command-line input arguments
    j_args = {
        "common": {
            "fsl_bin_path": cli_args["fsl_bin_path"]
        },

        "bibsnet": {
            "model": cli_args["nnUNet_configuration"],
            "nnUNet_predict_path": cli_args["nnUNet"]
        }
    }
    script_dir_attr = as_cli_attr(SCRIPT_DIR_ARG)
    j_args["meta"] = {script_dir_attr: SCRIPT_DIR,
                      "slurm": bool(cli_args[script_dir_attr])}

    # Crash immediately if the end is given as a stage that happens before start
    if (stage_names.index(cli_args["start"])
            > stage_names.index(cli_args["end"])):
        parser.error("Error: {} stage must happen before {} stage."
                     .format(cli_args["start"], cli_args["end"]))

    # Add command-line arguments to j_args
    j_args["stage_names"] = {"start": cli_args["start"],
                             "end": cli_args["end"]}  # TODO Maybe save the stage_names list in here too to replace optional_out_dirs use cases?
    for arg_to_add in ("bids_dir", "overwrite", "verbose", "work_dir"):
        j_args["common"][arg_to_add] = cli_args[arg_to_add]

    # TODO Remove all references to the optional_out_dirs arguments, and change
    #      j_args[optional_out_dirs][derivatives] to instead be j_args[common][output_dir]
    j_args["optional_out_dirs"] = {stagename: None for stagename in stage_names} 
    j_args["optional_out_dirs"]["derivatives"] = cli_args["output_dir"]


    # Define (and create) default paths in derivatives directory structure for 
    # each stage of each session of each subject
    sub_ses_IDs = get_all_sub_ses_IDs(j_args, cli_args["participant_label"],
                                      cli_args["session"])  # TODO Add brain_z_size into j_args[ID]
    
    # TODO Iff the user specifies a session, then let them specify an age
    default_derivs_dir = os.path.join(j_args["common"]["bids_dir"], "derivatives")
    for ix in range(len(sub_ses_IDs)):
        # Create a list with the subject ID and (if it exists) the session ID
        sub_ses = [sub_ses_IDs[ix]["subject"]]
        if sub_ses_IDs[ix].get("session"): 
            sub_ses.append(sub_ses_IDs[ix]["session"])

        j_args = ensure_j_args_has_bids_subdirs(
            j_args, stage_names, sub_ses, default_derivs_dir
        )

        # Verify that subject-session BIDS/input directory path exists
        sub_ses_dir = os.path.join(j_args["common"]["bids_dir"], *sub_ses)
        if not os.path.exists(sub_ses_dir):
            parser.error("No subject/session directory found at {}\nCheck "
                         "that your participant_label and session are correct."
                         .format(sub_ses_dir))
    
        # User only needs sessions.tsv if they didn't specify age_months
        if not j_args["common"].get("age_months"): 
            sub_ses_IDs[ix]["age_months"] = read_from_tsv(
                j_args, "age", *sub_ses
            )
        
        # Infer brain_z_size for this sub_ses using sessions.tsv if the 
        # user said to (by using --brain-z-size flag), otherwise infer it 
        # using age_months and the age-to-head-radius table .csv file
        sub_ses_IDs[ix]["brain_z_size"] = read_from_tsv(
                j_args, "brain_z_size", *sub_ses
            ) if cli_args["brain_z_size"] else get_brain_z_size(
                sub_ses_IDs[ix]["age_months"], j_args)


        # Check whether this sub ses has T1w and/or T2w input data
        data_path_BIDS_T = dict()  # Paths to expected input data to check
        for t in (1, 2):
            data_path_BIDS_T[t] = os.path.join(j_args["common"]["bids_dir"],
                                               *sub_ses, "anat",
                                               f"*T{t}w.nii.gz")
            sub_ses_IDs[ix][f"has_T{t}w"] = bool(glob(data_path_BIDS_T[t]))

        models_df = get_df_with_valid_bibsnet_models(sub_ses_IDs[ix])
        sub_ses_IDs[ix]["model"] = validate_model_num(
            cli_args, data_path_BIDS_T, models_df, sub_ses_IDs[ix], parser
        )

        # Create BIBSnet in/out directories
        dir_BIBSnet = dict()
        for io in ("in", "out"):
            dir_BIBSnet[io] = os.path.join(j_args["optional_out_dirs"]["bibsnet"],
                                           *sub_ses, f"{io}put")
            os.makedirs(dir_BIBSnet[io], exist_ok=True)

    if j_args["common"]["verbose"]:
        LOGGER.info(" ".join(sys.argv[:]))  # Print all

    # 2. roi2full for preBIBSnet and postBIBSnet transformation
    # j_args["xfm"]["roi2full"] =   # TODO
    return j_args, sub_ses_IDs


def ensure_j_args_has_bids_subdirs(j_args, derivs, sub_ses, default_parent):
    """
    :param j_args: Dictionary containing all args
    :param derivs: Unpacked list of strings. Each names 1 part of a path under
                   j_args[common][bids_dir]. The last string is mapped by
                   j_args[optional_out_dirs] to the subdir path.
    :param sub_ses: List with either only the subject ID str or the session too
    :param default_parent: The default parent directory where all output
                   directories will be placed.
    :return: j_args, but with the (now-existing) subdirectory path
    """

    j_args["optional_out_dirs"] = make_given_or_default_dir(
        j_args["optional_out_dirs"], "derivatives", default_parent
    )
    work_dir_list = ["prebibsnet", "bibsnet", "postbibsnet"]
    for deriv in derivs:
        subdir_path = os.path.join(j_args["common"]["work_dir"], deriv) if deriv in work_dir_list else os.path.join(
                j_args["optional_out_dirs"]["derivatives"], deriv)
        j_args["optional_out_dirs"] = make_given_or_default_dir(
            j_args["optional_out_dirs"], deriv, subdir_path
        )
        os.makedirs(os.path.join(j_args["optional_out_dirs"][deriv], *sub_ses),
                    exist_ok=True)  # Make all subject-session output dirs
    return j_args


def get_df_with_valid_bibsnet_models(sub_ses_ID):
    """
    :param sub_ses_ID: Dictionary mapping subject-session-specific input
                       parameters' names (as strings) to their values for
                       this subject session; the same as j_args[ID]
    :return: pandas.DataFrame of all bibsnet models viable for the input data
    """
    # Read in models.csv info mapping model num to which T(s) it has
    models_df = pd.read_csv(os.path.join(SCRIPT_DIR, "data", "models.csv"))

    # Exclude any models which require (T1w or T2w) data the user lacks
    for t in only_Ts_needed_for_bibsnet_model(sub_ses_ID):
        models_df = select_model_with_data_for_T(
            t, models_df, sub_ses_ID[f"has_T{t}w"]
        )
    return models_df


def get_all_sub_ses_IDs(j_args, subj_or_none, ses_or_none):
    """
    :param j_args: Dictionary containing all args
    :param subj_or_none: String (the subject ID) or a falsey value
    :param ses_or_none: String (the session name) or a falsey value
    :return: List of dicts; each dict maps "subject" to its subject ID string
             and may also map "session" to its session ID string
    """
    sub_ses_IDs = list()

    # Find all subject-session directories in tree under bids_dir
    sub_match = "sub-{}".format(subj_or_none.split("sub-")[-1] if
                                subj_or_none else "*")
    ses_match = "ses-{}".format(ses_or_none.split("ses-")[-1] if
                                ses_or_none else "*")
    sub_ses_dirs = glob(os.path.join(j_args["common"]["bids_dir"],
                        sub_match, ses_match))

    # If there are subjects and sessions, then add each pair to a list
    if sub_ses_dirs:
        for sub_ses_dirpath in sub_ses_dirs:
            sub_dirpath, session = os.path.split(sub_ses_dirpath)
            _, subject = os.path.split(sub_dirpath)
            sub_ses_IDs.append({"subject": subject, "session": session})

    # Otherwise, make a list that only has subject IDs
    else:
        for sub_dirpath in glob(os.path.join(j_args["common"]["bids_dir"],
                                sub_match)):
            sub_ses_IDs.append({"subject": os.path.basename(sub_dirpath)})

    return sub_ses_IDs


def get_brain_z_size(age_months, j_args, buffer=5):
    """ 
    Infer a participant's brain z-size from their age and from the average
    brain diameters table at the AGE_TO_HEAD_RADIUS_TABLE path
    :param age_months: Int, participant's age in months
    :param j_args: Dictionary containing all args
    :param buffer: Int, extra space (in mm), defaults to 5
    :return: Int, the brain z-size (height) in millimeters
    """
    MM_PER_IN = 25.4  # Conversion factor: inches to millimeters

    # Other columns' names in the age-to-head-radius table
    age_months_col = "Candidate_Age(mo.)"
    head_r_col = "Head_Radius(in.)"
    head_diam_mm = "head_diameter_mm"

    # Get table mapping each age in months to average head radius
    age2headradius = pd.read_csv(AGE_TO_HEAD_RADIUS_TABLE)

    # Get BCP age (in months) closest to the subject's age
    closest_age = get_age_closest_to(age_months, age2headradius[age_months_col])
    if j_args["common"]["verbose"]:
        LOGGER.info(f"Subject age in months: {age_months}\nClosest BCP age in "
                    f"months in age-to-head-radius table: {closest_age}")

    # Get average head radii in millimeters by age from table
    age2headradius[head_diam_mm] = age2headradius[head_r_col
                                                  ] * MM_PER_IN * 2
    row = age2headradius[age2headradius[age_months_col] == closest_age]
    
    # Return the average brain z-size for the participant's age
    return math.ceil(row.get(head_diam_mm)) + buffer


def read_from_tsv(j_args, col_name, *sub_ses):
    """
    :param j_args: Dictionary containing all args
    :param col_name: String naming the column of sessions.tsv to return
                     a value from (for this subject or subject-session)
    :param sub_ses: Tuple containing subject and session labels. 
    :return: Int, either the subject's age (in months) or the subject's
             brain_z_size (depending on col_name) as listed in sessions.tsv
    """

    session_tsv_path = os.path.join(j_args["common"]["bids_dir"], sub_ses[0],
                     "{}_sessions.tsv".format(sub_ses[0]))
    participant_tsv_path = os.path.join(j_args["common"]["bids_dir"],
                     "participants.tsv")

    ID_col = "session_id" if os.path.exists(session_tsv_path) else "participant_id"

    tsv_path = session_tsv_path if ID_col == "session_id" else participant_tsv_path

    tsv_df = pd.read_csv(
        tsv_path, delim_whitespace=True, index_col=ID_col
    )
    # Check if column name exists in either tsv, grab the value if column name exists
    try:
        if col_name not in tsv_df.columns:
            raise ValueError("Did not find {} in {}".format(col_name, tsv_path))
        else:
            col_value = get_col_value_from_tsv(j_args, tsv_df, ID_col, col_name, sub_ses)
    except ValueError as exception:
        LOGGER.info(exception)
        if ID_col == "participant_id":
            pass 
        else:
            ID_col = "participant_id"
            tsv_path = participant_tsv_path
            tsv_df = pd.read_csv(
                tsv_path, delim_whitespace=True, index_col=ID_col
            )
            if col_name not in tsv_df.columns:
                raise ValueError("Did not find {} in {}".format(col_name, tsv_path))
            else:
                col_value = get_col_value_from_tsv(j_args, tsv_df, ID_col, col_name, sub_ses)
        
    return col_value


def get_col_value_from_tsv(j_args, tsv_df, ID_col, col_name, sub_ses):
    # Get and return the col_name value from sessions.tsv

    subj_row = tsv_df.loc[
        ensure_prefixed(sub_ses[1], "ses-") if ID_col == "session_id" else ensure_prefixed(sub_ses[0], "sub-")
    ]  # select where "participant_id" matches
    if j_args["common"]["verbose"]:
        LOGGER.info(f"ID_col used to get details from tsv: {ID_col}")
        LOGGER.info(f"Subject details from tsv row:\n{subj_row}")
    return int(subj_row[col_name])


def validate_model_num(cli_args, data_path_BIDS_T, models_df, sub_ses_ID, parser):
    """
    :param cli_args: Dictionary containing all command-line input arguments
    :param data_path_BIDS_T: Dictionary mapping 1 and 2 to the (incomplete)
                             paths to expected T1w and T2w data respectively
    :param models_df: pd.DataFrame of all bibsnet models viable for input data
    :param sub_ses_ID: Dict mapping (string) names to values for sub- &
                       ses-specific input parameters; same as j_args[ID]
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :return: Int, validated bibsnet model number
    """
    model = cli_args["model"]  # Model number (if given from command line)

    # Exclude any models which require (T1w or T2w) data the user lacks
    for t in (1, 2):

        # If user gave a model number but not the data the model needs,
        # then crash with an informative error message
        if model and (model not in models_df["model_num"]):
            parser.error("CABINET needs T{}w data at the path below " 
                            "to run model {}, but none was found.\n{}\n"
                            .format(t, model, data_path_BIDS_T[t]))

    if not model:  # Get default model number if user did not give one
        models_df = models_df[models_df["is_default"]]
        if len(models_df) > 1:
            for t in (1, 2):
                models_df = select_model_with_data_for_T(
                    t, models_df, sub_ses_ID[f"has_T{t}w"]
                )
        model = models_df.squeeze()["model_num"]
            
    return model


def select_model_with_data_for_T(t, models_df, has_T):
    """
    :param t: Int, either 1 or 2 (to signify T1w or T2w respectively)
    :param models_df: pandas.DataFrame with columns called "T1w" and "T2w"
                      with bool values describing which T(s) a model needs
    :param has_T: bool, True if T{t}w data exists for this subject/ses
    :return: pandas.DataFrame, all models_df rows with data for this sub/ses/t
    """
    has_T_row = models_df[f"T{t}w"]
    return models_df.loc[has_T_row if has_T else ~has_T_row]


def as_cli_attr(cli_arg_str):
    """
    :param cli_arg_str: String in command-line argument form
    :return: cli_arg_str, but formatted as a stored command-line argument
    """
    return cli_arg_str.strip("-").replace("-", "_")


def make_given_or_default_dir(dirs_dict, dirname_key, default_dirpath):
    """
    :param dirs_dict: Dictionary which must map dirname_key to a valid path
    :param dirname_key: String which dirs_dict must map to a valid path
    :param default_dirpath: String, valid directory path to map dirname_key to
                            unless dirname_key's already mapped to another path
    :return: dirs_dict, but with dirname_key mapped to a valid directory path
    """
    dirs_dict = ensure_dict_has(dirs_dict, dirname_key, default_dirpath)
    os.makedirs(dirs_dict[dirname_key], exist_ok=True)
    return dirs_dict


def ensure_dict_has(a_dict, a_key, new_value):
    """
    :param a_dict: Dictionary (any)
    :param a_key: Object which will be a key in a_dict
    :param new_value: Object to become the value mapped to a_key in a_dict
                      unless a_key is already mapped to a value
    :return: a_dict, but with a_key mapped to some value
    """
    if not dict_has(a_dict, a_key):
        a_dict[a_key] = new_value
    return a_dict

