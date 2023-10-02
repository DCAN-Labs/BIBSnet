import argparse
import os

from src.utilities import ensure_prefixed


def always_true(*_):
    """
    This function is useful when e.g. every type except 1 has a corresponding
    input validation function, because this function is used for that extra 1 
    :return: True, regardless of what the input arguments are 
    """
    return True


def valid_float_0_to_1(val):
    """
    :param val: Object to check, then throw an error if it is invalid
    :return: val if it is a float between 0 and 1 (otherwise invalid)
    """
    return validate(val, lambda x: 0 <= float(x) <= 1, float,
                    "{} is not a number between 0 and 1")


def valid_output_dir(path):
    """
    Try to make a folder for new files at path; throw exception if that fails
    :param path: String which is a valid (not necessarily real) folder path
    :return: String which is a validated absolute path to real writeable folder
    """
    return validate(path, lambda x: os.access(x, os.W_OK),
                    valid_readable_dir, "Cannot create directory at {}",
                    lambda y: os.makedirs(y, exist_ok=True))


def valid_output_dir_or_none(path):
    """
    Try to make a folder for new files at path, unless "path" is just False.
    Throw exception if that fails
    :param path: String which should be either a valid (not necessarily real)
                 folder path or None
    :return: Either None or a validated absolute path to real writeable folder
    """
    return path if path is None else valid_output_dir(path)


def valid_positive_float(to_validate):
    """
    Throw argparse exception unless to_validate is a positive float
    :param to_validate: Object to test whether it is a positive float
    :return: to_validate if it is a positive float
    """
    return validate(to_validate, lambda x: float(x) >= 0, float,
                    "{} is not a positive number")


def valid_readable_dir(path):
    """
    :param path: Parameter to check if it represents a valid directory path
    :return: String representing a valid directory path
    """
    return validate(path, os.path.isdir, valid_readable_file,
                    "Cannot read directory at '{}'")


def valid_readable_file(path):
    """
    Throw exception unless parameter is a valid readable filepath string. Use
    this, not argparse.FileType("r") which leaves an open file handle.
    :param path: Parameter to check if it represents a valid filepath
    :return: String representing a valid filepath
    """
    return validate(path, lambda x: os.access(x, os.R_OK),
                    os.path.abspath, "Cannot read file at '{}'")


def valid_readable_json(path):
    """
    :param path: Parameter to check if it represents a valid .json file path
    :return: String representing a valid .json file path
    """
    return validate(path, lambda _: os.path.splitext(path)[-1] == ".json",
                    valid_readable_file,
                    "'{}' is not a path to a readable .json file")


def valid_subj_ses(in_arg, pfx, name): 
    """
    :param in_arg: Object to check if it is a valid subject ID or session name
    :param pfx: String that's the prefix to an ID; "sub-" or "ses-"
    :param name: String describing what in_arg should be (e.g. "subject")
    :return: True if in_arg is a valid subject ID or session name; else False
    """
    return validate(in_arg, always_true, lambda x: ensure_prefixed(x, pfx),
                    "'{}'" + " is not a valid {}".format(name))


def valid_whole_number(to_validate):
    """
    Throw argparse exception unless to_validate is a positive integer
    :param to_validate: Object to test whether it is a positive integer
    :return: to_validate if it is a positive integer
    """
    return validate(to_validate, lambda x: int(x) >= 0, int,
                    "{} is not a positive integer")


def valid_subj_ses_ID(to_validate):
    """
    :param to_validate: Object to turn into a valid subject/session ID label
    :return: String, valid subject/session ID label
    """  # TODO Validate that subject/session exists 
    return validate(to_validate, always_true, lambda x: x.split("-")[-1],
                    "{} is not a valid subject/session ID.")


def validate(to_validate, is_real, make_valid, err_msg, prepare=None):
    """
    Parent/base function used by different type validation functions. Raises an
    argparse.ArgumentTypeError if the input object is somehow invalid.
    :param to_validate: String to check if it represents a valid object 
    :param is_real: Function which returns true iff to_validate is real
    :param make_valid: Function which returns a fully validated object
    :param err_msg: String to show to user to tell them what is invalid
    :param prepare: Function to run before validation
    :return: to_validate, but fully validated
    """
    try:
        if prepare:
            prepare(to_validate)
        assert is_real(to_validate)
        return make_valid(to_validate)
    except (OSError, TypeError, AssertionError, ValueError,
            argparse.ArgumentTypeError):
        raise argparse.ArgumentTypeError(err_msg.format(to_validate))


def validate_parameter_types(j_args, j_types, param_json, parser, stage_names):
    """
    Verify that every parameter in j_args is the correct data-type and the 
    right kind of value. If any parameter is invalid, crash with an error.
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param j_types: Dictionary mapping every argument in j_args to its type
    :param param_json: String, path to readable .JSON file with all parameters
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    :param stage_names: List of strings; each names a stage to run
    """
    # Define functions to validate arguments of each data type
    type_validators = {"bool": bool, "int": int,
                       "existing_directory_path": valid_readable_dir,
                       "existing_file_path": valid_readable_file,
                       "existing_json_file_path": valid_readable_json,
                       "float_0_to_1": valid_float_0_to_1,
                       "new_directory_path": valid_output_dir,
                       "new_file_path": always_true,  # TODO Make "valid_output_filename" function to ensure that filenames don't have spaces or slashes, and maaaaybe to ensure that the new file's parent directory exists?
                       "optional_new_dirpath": valid_output_dir_or_none,
                       "optional_real_dirpath": valid_output_dir_or_none,
                       "positive_float": valid_positive_float,
                       "positive_int": valid_whole_number, 
                       "str": always_true}

    required_for_stage = {
        "nibabies": ["cifti_output", "fd_radius", "work_dir"],
        "xcpd": ["cifti", "combineruns", "fd_thresh",
                 "head_radius", "input_type"]
    }

    # Get a list of all stages after the last stage to run
    after_end = stage_names[stage_names.index(j_args["stage_names"]["end"])+1:]

    # Verify parameters in each section
    to_delete = list()
    for section_orig_name, section_dict in j_types.items():
        section_name = section_orig_name.lower()  # TODO Should we change parameter types .JSON file to make section names already lowercase?

        # Skip the j_args sections for stages not being run
        if section_name in stage_names and section_name in after_end:
            if section_orig_name in j_args:
                to_delete.append(section_orig_name)

        # Only include resource_management if we're in SLURM/SBATCH job(s)
        elif not (section_name == "resource_management"
                  and not j_args["meta"]["slurm"]):

            # Validate every parameter in the section
            for arg_name, arg_type in section_dict.items():

                # Ignore XCP-D and nibabies parameters that are null
                arg_value = j_args[section_name][arg_name]
                if not (arg_value is None and
                        section_name in ("nibabies", "XCPD") and
                        arg_value not in required_for_stage[section_name]):
                    validate_1_parameter(j_args, arg_name, arg_type, section_name,
                                         type_validators, param_json, parser)

    # Remove irrelevant parameters
    for section_name in to_delete:
        del j_args[section_name]
    return j_args


def validate_1_parameter(j_args, arg_name, arg_type, section_name,
                         type_validators, param_json, parser):
    """
    :param j_args: Dictionary containing all args from parameter .JSON file
    :param arg_name: String naming the parameter to validate
    :param arg_type: Either a string naming the data type of the parameter to 
                     validate or a list of options the parameter must be in
    :param section_name: String that's a subcategory in the param_json file
    :param type_validators: Dict mapping each arg_type to a validator function
    :param param_json: String, path to readable .JSON file with all parameters
    :param parser: argparse.ArgumentParser to raise error if anything's invalid
    """
    to_validate = j_args[section_name][arg_name]  # Parameter to validate
    err_msg = ("'{}' is not a valid '{}' parameter in the '{}' section "
               "of {} (Problem: {})")  # Message for if to_validate is invalid
    try:
        # Run a type validation function unless arg_type is a list
        if isinstance(arg_type, str):
            type_validators[arg_type](to_validate)
            
        # Verify that the parameter is a valid member of a choices list
        elif isinstance(arg_type, list) and to_validate not in arg_type:
            parser.error(
                err_msg.format(to_validate, arg_name, section_name, param_json,
                               "Valid {} values: {}"
                               .format(arg_name, ", ".join(arg_type)))
            )

    # If type validation fails then inform the user which parameter
    # has an invalid type and what the valid types are
    except (argparse.ArgumentTypeError, KeyError, TypeError, ValueError) as e:
        parser.error(err_msg.format(to_validate, arg_name,
                                    section_name, param_json, e))

