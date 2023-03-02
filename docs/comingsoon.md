# Coming Soon

### Multiple Anatomical Images of the Same Modality in the BIDS Inputs

At present, CABINET is unable to operate if there are two or more T1w or T2w images present in the BIDS input anat directories. To use CABINET (and subsequent processing steps), it is advisable to eliminate the lower quality run from the BIDS inputs. We plan to add functionality to select the first T1w or T2w image in the near future, with averaging implemented at a later stage.

### BIDS Validation

We will be integrating the BIDS validator into CABINET in the near future to ensure that all inputs are valid. This enhancement will save the user time by catching any BIDS input errors immediately that CABINET previously would not interact with (func, fmap, and dwi directories) and ensuring the user will not have to go back and BIDSify a database after running CABINET and before running a subsequent processing stage. This will ensure consistency between inputs of both stages.

### Nibabies and XCP-D Incorporation

CABINET intends to include Nibabies and XCP-D within its stages in the future in order to be a more efficient, flexible, and reproducible pipeline.
