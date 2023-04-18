## Participant and Sessions TSV Requirements

CABINET requires information on the age of the participant you are processing. If you have more than one subject, the codebase uses the `sub-{}_sessions.tsv` inside the BIDS directory at the session-level for the age. If you have a single session or no sessions specified, the codebase will use the `participants.tsv` inside the BIDS directory at the subject-level or the `sub-{}_sessions.tsv` inside the BIDS directory at the session-level for the age. While not required, if you wish to specify a brain_z_size for your participants, the codebase will search in the same location as it does for the age.

Below we go over what each TSV file should contain.

### Format Specification Example

`sub-{}_sessions.tsv`:

| session_id | age |
|:-:|:-:|
| ses-A | 1 |

`participants.tsv`:

| subject | session_id | age |
|:-:|:-:|:-:|
| sub-01 | ses-A | 1 |

NOTE: the `ses-` prefix is currently required for `session_id` values.

### Content

When running multiple subjects and/or sessions, the `sub-{}_sessions.tsv` file in each subject's directory (at the session directory level) must include an `age` column. In that column, each row has one positive integer, the participant's age in months at that session.

<br />
<img src="https://user-images.githubusercontent.com/102316699/184005162-0b1ebb76-3e5a-4bd3-b258-a686272e2ecc.png" width=555em style="margin-left: auto; margin-right: auto; display: block" />
<br />

If the user wants to specify the brain height (shown above) for each subject session, then the user must also include an additional `"brain_z_size"` column. That column also must have a positive integer for each row: the size of the participant's brain along the z-axis in millimeters. The TSV file should look like the examples below:

`sub-{}_sessions.tsv`

| session_id | age | brain_z_size |
|:-:|:-:|:-:|
| ses-X | 1 | 120 |
| ses-X | 6 | 145 |

`participants.tsv`:

| subject | session_id | age | brain_z_size |
|:-:|:-:|:-:|:-:|
| sub-01 | ses-A | 1 | 130 |

Without a `brain_z_size` column, `CABINET` will calculate the `brain_z_size` value based on a table with [BCP](https://babyconnectomeproject.org/) participants' average head radius per age. That table is called `age_to_avg_head_radius_BCP.csv` under the `data` directory.
