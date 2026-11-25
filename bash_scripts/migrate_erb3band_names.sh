#!/bin/bash
# Rename / move everything of the ERB 3-band sweep design to its design name, sweep_design_erb3band, so that all
# designs follow one rule: everything of a design lives under its name.
#
#   stimuli/produced/sweep_design_erb_3band            -> stimuli/produced/sweep_design_erb3band
#   stimuli/produced/sweep_design_erb_3band_<variant>  -> stimuli/produced/sweep_design_erb3band_<variant>
#   models_output/sweep_design_erb_3band[_<variant>]   -> models_output/sweep_design_erb3band[_<variant>]
#   sweep_recovery_results/<everything>                -> sweep_recovery_results/sweep_design_erb3band/<everything>
#   sweep_groundtruth_results/<everything>             -> sweep_groundtruth_results/sweep_design_erb3band/<everything>
#   sweep_recovery_multirun_grids/<everything>         -> sweep_recovery_multirun_grids/sweep_design_erb3band/<everything>
#   (a folder sweep_recovery_multirun_grids that sits INSIDE sweep_recovery_results/ becomes
#    sweep_recovery_multirun_grids/sweep_design_erb3band, i.e. the design's grid folder)
#
# Safe to run twice: it only moves what is still under the old names, never overwrites (a name that already exists
# is reported as a CONFLICT and skipped) and never deletes. It changes nothing unless you pass --apply.
# Moves stay on one file system, so they are instant whatever the size.
#
# Usage, from the project root (the same on your laptop and on dIPC):
#   bash bash_scripts/migrate_erb3band_names.sh            # dry run: prints what it WOULD do
#   bash bash_scripts/migrate_erb3band_names.sh --apply    # does it
# On dIPC the project root is /scratch/ecelikay/workspace/auditory-pRF-subcortical. Run it on the login node (it is
# only mv). Do it AFTER copying the updated code there and BEFORE submitting new jobs.

set -u
DESIGN=sweep_design_erb3band
OLD=sweep_design_erb_3band
KNOWN_DESIGNS=" sweep_design_erb3band harmonic_sweep_armA harmonic_sweep_armB "
APPLY=0
if [ "${1:-}" = "--apply" ]; then APPLY=1; fi

if [ ! -d auditory_prf ] || [ ! -d stimuli ]; then
    echo "Run this from the project root (the folder that contains auditory_prf/ and stimuli/)." >&2
    exit 2
fi

N_MOVED=0
N_CONFLICT=0

do_move() {   # do_move <from> <to>
    local from="$1" to="$2"
    if [ -e "$to" ]; then
        echo "CONFLICT (skipped, target exists): $from -> $to"
        N_CONFLICT=$((N_CONFLICT + 1))
        return
    fi
    if [ "$APPLY" -eq 1 ]; then
        mkdir -p "$(dirname "$to")"
        mv "$from" "$to"
        echo "moved: $from -> $to"
    else
        echo "would move: $from -> $to"
    fi
    N_MOVED=$((N_MOVED + 1))
}

rename_old_design_folders() {   # <root>: sweep_design_erb_3band* -> sweep_design_erb3band*
    local root="$1" path name
    for path in "$root"/${OLD}*; do
        [ -d "$path" ] || continue
        name=$(basename "$path")
        do_move "$path" "$root/${name/$OLD/$DESIGN}"
    done
}

move_under_design() {   # <root>: every entry that is not a design folder -> <root>/sweep_design_erb3band/
    local root="$1" path name
    [ -d "$root" ] || return
    for path in "$root"/* "$root"/.[!.]*; do
        [ -e "$path" ] || continue
        name=$(basename "$path")
        case "$KNOWN_DESIGNS" in *" $name "*) continue ;; esac
        if [ "$name" = "sweep_recovery_multirun_grids" ] && [ "$root" != "sweep_recovery_multirun_grids" ]; then
            do_move "$path" "sweep_recovery_multirun_grids/$DESIGN"     # a grids folder found inside another root
            continue
        fi
        do_move "$path" "$root/$DESIGN/$name"
    done
}

echo "ERB 3-band design -> $DESIGN  ($([ "$APPLY" -eq 1 ] && echo APPLY || echo "dry run, nothing is changed"))"
rename_old_design_folders stimuli/produced
rename_old_design_folders models_output
move_under_design sweep_recovery_results
move_under_design sweep_groundtruth_results
move_under_design sweep_recovery_multirun_grids

echo "----"
echo "$N_MOVED to move/moved, $N_CONFLICT conflicts"
if [ "$APPLY" -eq 0 ] && [ "$N_MOVED" -gt 0 ]; then
    echo "Nothing was changed. If the list above is right, run it again with --apply."
fi
