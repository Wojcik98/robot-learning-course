# Lab 2 exercise guide

The second training episode has a complete sequence when its frame indices equal `0..N-1`. Its stored timestamp step should be near 0.1 s because the dataset declares 10 Hz. Check the actual value in your run; do not infer the 20 Hz simulator controller period from it.

A relative action gives a change for the next control step. An absolute target names a position or pose. If a converter changes the meaning without changing the values, training pairs point to the wrong target. Record the action convention, coordinate frame, units, gripper meaning, and time base in the conversion manifest. A small HDF5 or RLDS-like record is only a storage example until these fields are checked.

The two new development states must stay in the same order, with each result present even if it failed. The saved reference reports 19 successes in 20 bowl-plate development starts. Two starts give too little information for a reliable success estimate. Neither result includes final states.
