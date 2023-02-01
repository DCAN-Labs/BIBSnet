## Multiple Participant Requirements

### `participants.tsv`

#### Format Specification Example

| participant_id | session | age |
|:-:|:-:|:-:|
| sub-123456 | ses-A | 1 |

NOTE: `sub-` and `ses-` prefixes are currently required for `participant_id` and `session` values.

#### Content

When running multiple subjects and/or sessions, the `participants.tsv` file in the `bids_dir` must include an `age` column. In that column, each row has one positive integer, the participant's age in months at that session.

<br />
<img src="https://user-images.githubusercontent.com/102316699/184005162-0b1ebb76-3e5a-4bd3-b258-a686272e2ecc.png" width=555em style="margin-left: auto; margin-right: auto; display: block" />
<br />

If the user wants to specify the brain height (shown above) for each subject session, then the user must also include an additional `"brain_z_size"` column. That column also must have a positive integer for each row: the size of the participant's brain along the z-axis in millimeters. The `participants.tsv` file should look like the example below:

| participant_id | session | age | brain_z_size |
|:-:|:-:|:-:|:-:|
| sub-123456 | ses-X | 1 | 120 |
| sub-234567 | ses-X | 6 | 145 |

Without a `brain_z_size` column, `CABINET` will calculate the `brain_z_size` value based on a table with [BCP](https://babyconnectomeproject.org/) participants' average head radius per age. That table is called `age_to_avg_head_radius_BCP.csv` under the `data` directory.
