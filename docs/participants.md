## Multiple Participant Requirements

`sub-{}_sessions.tsv`

### Format Specification Example

| session | age |
|:-:|:-:|
| ses-A | 1 |

NOTE: the `ses-` prefix is currently required for `session` values.

### Content

When running multiple subjects and/or sessions, the `sub-{}_sessions.tsv` file in each subject's directory (at the session directory level) must include an `age` column. In that column, each row has one positive integer, the participant's age in months at that session.

<br />
<img src="https://user-images.githubusercontent.com/102316699/184005162-0b1ebb76-3e5a-4bd3-b258-a686272e2ecc.png" width=555em style="margin-left: auto; margin-right: auto; display: block" />
<br />

If the user wants to specify the brain height (shown above) for each subject session, then the user must also include an additional `"brain_z_size"` column. That column also must have a positive integer for each row: the size of the participant's brain along the z-axis in millimeters. The `sub-{}_sessions.tsv` file for a given subject should look like the example below:

| session | age | brain_z_size |
:-:|:-:|:-:|
| ses-X | 1 | 120 |
| ses-X | 6 | 145 |

Without a `brain_z_size` column, `CABINET` will calculate the `brain_z_size` value based on a table with [BCP](https://babyconnectomeproject.org/) participants' average head radius per age. That table is called `age_to_avg_head_radius_BCP.csv` under the `data` directory.
