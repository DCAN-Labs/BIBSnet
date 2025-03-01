# FSL labels required for infant pipeline:
2   Left-Cerebral-White-Matter              245 245 245 0 (**NOTE: depending on which JLF templates are used, this label may be replace by separate labels for myelinated and unmyelinated WM**)
3   Left-Cerebral-Cortex                    205 62  78  0
4   Left-Lateral-Ventricle                  120 18  134 0
8   Left-Cerebellum-Cortex                  230 148 34  0
10  Left-Thalamus-Proper                    0   118 14  0
11  Left-Caudate                            122 186 220 0
12  Left-Putamen                            236 13  176 0
13  Left-Pallidum                           12  48  255 0
14  3rd-Ventricle                           204 182 142 0
15  4th-Ventricle                           42  204 164 0
16  Brain-Stem                              119 159 176 0
17  Left-Hippocampus                        220 216 20  0
18  Left-Amygdala                           103 255 255 0
24  CSF                                     60  60  60  0
26  Left-Accumbens-area                     255 165 0   0
28  Left-VentralDC                          165 42  42  0
41  Right-Cerebral-White-Matter             0   225 0   0 (**NOTE: depending on which JLF templates are used, this label may be replace by separate labels for myelinated and unmyelinated WM**)
42  Right-Cerebral-Cortex                   205 62  78  0
43  Right-Lateral-Ventricle                 120 18  134 0
47  Right-Cerebellum-Cortex                 230 148 34  0
49  Right-Thalamus-Proper                   0   118 14  0
50  Right-Caudate                           122 186 220 0
51  Right-Putamen                           236 13  176 0
52  Right-Pallidum                          13  48  255 0
53  Right-Hippocampus                       220 216 20  0
54  Right-Amygdala                          103 255 255 0
58  Right-Accumbens-area                    255 165 0   0
60  Right-VentralDC                         165 42  42  0
172 Vermis                                  119 100 176 0

#Labels specific to infants:
#Label 2 from above (left WM) is reassigned to labels 159 & 161
#Label 41 (right WM) is reassigned to labels 160 & 162

159 Left-Cerebral-WM-unmyelinated           
160 Right-Cerebral-WM-unmyelinated   
161 Left-Cerebral-WM-myelinated            
162 Right-Cerebral-WM-myelinated           
