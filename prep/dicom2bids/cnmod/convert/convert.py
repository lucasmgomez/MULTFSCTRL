from heudiconv.main import workflow as heudiconv_workflow
from argparse import ArgumentParser
import pathlib
import multiprocessing
from functools import partial
import fileinput
import shutil
from ..prepare.fill_intended_for import fill_intended_for, fill_b0_meta
import re

HEURISTICS_PATH = pathlib.Path(__file__).parent.resolve() / 'heuristics_unf.py'

def fix_scans_tsv_prune_missing_local(path):
    """
    For each *_scans.tsv, drop rows whose 'filename' does not exist in the dataset.
    Also drops rows pointing to *scout* or *localizer* (if any slipped through).
    """
    import csv

    scans = list(path.glob('sub-*/ses-*/*_scans.tsv'))
    for tsv in scans:
        try:
            with open(tsv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                rows = list(reader)
            if not rows:
                continue
            header = rows[0]
            # find the 'filename' column (case-insensitive)
            try:
                fi = next(i for i, c in enumerate(header) if c.strip().lower() == 'filename')
            except StopIteration:
                # no filename column → nothing to do
                continue

            kept = [header]
            for r in rows[1:]:
                if not r or fi >= len(r):
                    continue
                rel = r[fi].lstrip('/')  # BIDS allows leading '/'
                p = (path / rel)
                # drop scouts/localizers if present
                lower = rel.lower()
                if 'scout' in lower or 'localizer' in lower:
                    continue
                if p.exists():
                    kept.append(r)

            if len(kept) != len(rows):
                with open(tsv, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerows(kept)
        except Exception:
            # don't fail the run over a scans.tsv tweak
            continue

def parse_args():
    docstr = ("""Example:
             convert.py ....""")
    parser = ArgumentParser(description=docstr)
    parser.add_argument(
        "--output-file",
        required=True,
        help="path to the dataset where to store the sessions",
    )

    parser.add_argument(
        "--ria-storage-remote",
        required=False,
        help="name of the ria storage remote",
    )
    
    parser.add_argument(
        '--files',
        nargs='+',
        required=True,
        type=pathlib.Path,
        help='Files (tarballs, dicoms) or directories containing files to '
             'process. Cannot be provided if using --dicom_dir_template.')

    parser.add_argument(
        '--nprocs',
        type=int,
        default=4,
        help='number of jobs to run in parallel with multiprocessing')

    parser.add_argument(
        "--b0-field-id",
        action="store_true",
        help="fill new BIDS B0FieldIdentifier instead of IntendedFor",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="run heudiconv with overwrite",
    )
    
    return parser.parse_args()

def detect_sub_ses(ds, commit='HEAD'):
    new_files = [(ds.pathobj / nf) for nf in ds.repo.call_git(['show','--name-only', commit,'--format=oneline']).split('\n')[1:]]
    for f in new_files:
        mtch = re.match(r".*/sub-([a-zA-Z0-9]+)/(ses-([a-zA-Z0-9]+))?.*", str(f))
        if mtch:
            grps = mtch.groups()
            return grps[0], grps[2]

def detect_sub_ses_local(path):
    # simple filesystem scan fallback to determine subject/session
    for sub in sorted(path.glob('sub-*')):
        if sub.is_dir():
            subject = sub.name.split('-', 1)[1]
            for ses in sorted(sub.glob('ses-*')):
                if ses.is_dir():
                    return subject, ses.name.split('-', 1)[1]
            return subject, None
    return None, None

def fix_fmap_phase_local(path):
    phase_glob = 'sub-*/ses-*/fmap/*_part-phase*'
    mag_glob = 'sub-*/ses-*/fmap/*_part-mag*'
    phase_fmaps = list(path.glob(phase_glob))
    mag_fmaps = list(path.glob(mag_glob))
    if not phase_fmaps and not mag_fmaps:
        return
    # remove phase fmap files
    for f in phase_fmaps:
        try:
            f.unlink()
        except Exception:
            pass
    # rename mag files
    for f in mag_fmaps:
        new = pathlib.Path(str(f).replace('_part-mag', ''))
        try:
            f.rename(new)
        except Exception:
            pass
    scans_tsvs = list(path.glob('sub-*/ses-*/*_scans.tsv'))
    for tsv in scans_tsvs:
        txt = []
        with open(tsv, 'r', encoding='utf-8') as fh:
            for line in fh:
                if not all([k in line for k in ['fmap/','_part-phase']]):
                    if 'fmap/' in line:
                        line = line.replace('_part-mag', '')
                    txt.append(line)
        with open(tsv, 'w', encoding='utf-8') as fh:
            fh.writelines(txt)

def fix_fmap_multiecho_local(path):
    # Make EchoN part of acq label, regardless of other tokens between acq-... and _echo-#
    # Examples matched:
    #   ..._task-foo_acq-bar_echo-1_run-01_...
    #   ..._acq-bar_echo-2_...
    #   ..._acq-barSomething_echo-3_...
    echo_glob = '**/fmap/*_echo-*'
    echo_files = list(path.glob(echo_glob))
    if not echo_files:
        return
    pattern = re.compile(
        r"^(.*/sub-.*(_ses-[^_]+))(_acq-([A-Za-z0-9]+))(.*?)(_echo-([0-9]+))(_.*)$"
    )
    for f in echo_files:
        m = pattern.match(str(f))
        if not m:
            continue
        new_f = m.expand(r"\1_acq-\4Echo\7\5\8")
        try:
            pathlib.Path(f).rename(pathlib.Path(new_f))
        except Exception:
            pass
    scans_tsvs = list(path.glob('**/*_scans.tsv'))
    for tsv in scans_tsvs:
        txt = []
        with open(tsv, 'r', encoding='utf-8') as fh:
            for line in fh:
                if not all([k in line for k in ['fmap/','_part-phase']]):
                    if 'fmap/' in line:
                        line = re.sub(
                            r"^(.*/sub-.*(_ses-[^_]+))(_acq-([A-Za-z0-9]+))(.*?)(_echo-([0-9]+))(_.*)$",
                            r"\1_acq-\4Echo\7\5\8",
                            line
                        )
                    txt.append(line)
        with open(tsv, 'w', encoding='utf-8') as fh:
            fh.writelines(txt)

def fix_complex_events_local(path):
    # remove phase event files and rename mag events
    phase_events = [f for f in path.glob('sub-*/ses-*/func/*_part-phase*_events.tsv')]
    for f in phase_events:
        try:
            f.unlink()
        except Exception:
            pass
    mag_events = [f for f in path.glob('sub-*/ses-*/func/*_part-mag*_events.tsv')]
    for f in mag_events:
        try:
            f.rename(pathlib.Path(str(f).replace('_part-mag', '')))
        except Exception:
            pass

def single_session_job(input_file, output_file, ria_storage_remote, b0_field_id=False, overwrite=False):
    session_name = input_file.stem.split('.')[0]
    remote_path = pathlib.Path(output_file.replace('ria+file://', '').replace('#~', '/alias/').split('@')[0])
    print("Output path: ", remote_path)
    try:
        # with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdirname:
        tmpdirname = f"/mnt/tempdata/lucas/fmri/tmp/dcm2niix_out/{session_name}"
        pathlib.Path(tmpdirname).mkdir(parents=True, exist_ok=True)

        # filesystem-only path: run heudiconv without datalad and operate on the tmp dir
        bids_opts = ['--overwrite'] if overwrite else []

        heudiconv_params = dict(
            files=[str(input_file)],
            outdir=tmpdirname,
            bids_options=bids_opts,
            datalad=False,
            heuristic=str(HEURISTICS_PATH),
        )
        heudiconv_workflow(**heudiconv_params)

        # --- AFTER heudiconv (still in the TEMP tree) ---
        tmp_path = pathlib.Path(tmpdirname)
        subject, session = detect_sub_ses_local(tmp_path)

        # cleanups in temp tree before copying
        fix_fmap_phase_local(tmp_path)
        fix_complex_events_local(tmp_path)
        fix_fmap_multiecho_local(tmp_path)

        # NEW: prune scans.tsv rows pointing to deleted/nonexistent files
        fix_scans_tsv_prune_missing_local(tmp_path)

        # --- COPY results from TEMP → FINAL ---
        try:
            dst = remote_path
            dst.mkdir(parents=True, exist_ok=True)
            for item in tmp_path.iterdir():
                if item.name == '.heudiconv':  # skip heudiconv cache
                    continue
                dest_item = dst / item.name
                if dest_item.exists():
                    if overwrite:
                        if dest_item.is_dir():
                            shutil.rmtree(dest_item)
                        else:
                            dest_item.unlink()
                    else:
                        print(f"Skipping existing path (use --overwrite to replace): {dest_item}")
                        continue
                if item.is_dir():
                    shutil.copytree(item, dest_item)
                else:
                    shutil.copy2(item, dest_item)
            print(f"Saved converted files to {dst}")
        except Exception as exc:
            print(f"Failed to copy outputs to {remote_path}: {exc}")

        # --- run metadata fillers ON THE FINAL TREE (once) ---
        target_path = dst
        if b0_field_id:
            fill_b0_meta(target_path, participant_label=subject, session_label=session)
        else:
            fill_intended_for(target_path, participant_label=subject, session_label=session)

        # (Also prune scans.tsv in the final tree, just in case:
        fix_scans_tsv_prune_missing_local(target_path)

        print(f"processed {input_file}")
    except Exception as e:
        print(f"An error occur processing {input_file}")
        print(e)
        import traceback
        print(traceback.format_exc())
        return traceback.format_exc()

def fix_fmap_phase(ds, commit=None):
    #### fix fmap phase data (sbref series will contain both and heudiconv auto name it)

    phase_glob = 'sub-*/ses-*/fmap/*_part-phase*'
    mag_glob = 'sub-*/ses-*/fmap/*_part-mag*'
    phase_fmaps = ds.pathobj.glob(phase_glob)
    mag_fmaps = ds.pathobj.glob(mag_glob)

    if commit:
        new_files = [(ds.pathobj / nf) for nf in ds.repo.call_git(['show','--name-only', commit,'--format=oneline']).split('\n')[1:]]
        phase_fmaps = [f for f in phase_fmaps if f in new_files]
        mag_fmaps = [f for f in mag_fmaps if f in new_files]
        scans_tsvs = [nf for nf in new_files if '_scans.tsv' in str(nf)]
    else:
        phase_fmaps = list(phase_fmaps)
        mag_fmaps = list(mag_fmaps)
        scans_tsvs = list(ds.pathobj.glob('sub-*/ses-*/*_scans.tsv'))
        
    if not phase_fmaps and not mag_fmaps:
        return
    ds.repo.remove(phase_fmaps)

    for f in mag_fmaps:
        ds.repo.call_git(['mv', str(f), str(f).replace('_part-mag','')])

    ds.unlock(scans_tsvs, on_failure='ignore')
    with fileinput.input(files=scans_tsvs, inplace=True) as f:
        for line in f:
            if not all([k in line for k in ['fmap/','_part-phase']]): # remove phase fmap
                if 'fmap/' in line:
                    line = line.replace('_part-mag', '')
                print(line, end='')
    ds.save(message='fix fmap phase')

def fix_fmap_multiecho(ds, commit=None):
    # Robust echo mover that does not assume any desc- tokens.
    echo_glob = '**/fmap/*_echo-*'
    echo_files = ds.pathobj.glob(echo_glob)
    if commit:
        new_files = [(ds.pathobj / nf) for nf in ds.repo.call_git(['show','--name-only', commit,'--format=oneline']).split('\n')[1:]]
        echo_files = [f for f in echo_files if f in new_files]
    else:
        echo_files = list(echo_files)
    
    if not list(echo_files):
        return

    pattern = re.compile(
        r"^(.*/sub-.*(_ses-[^_]+))(_acq-([A-Za-z0-9]+))(.*?)(_echo-([0-9]+))(_.*)$"
    )
    for f in echo_files:
        s = str(f)
        m = pattern.match(s)
        if not m:
            continue
        new_f = m.expand(r"\1_acq-\4Echo\7\5\8")
        ds.repo.call_git(['mv', s, new_f])

    if commit:
        scans_tsvs = [nf for nf in new_files if '_scans.tsv' in str(nf)]
    else:
        scans_tsvs = list(ds.pathobj.glob('**/*_scans.tsv'))
        
    ds.unlock(scans_tsvs, on_failure='ignore')
    with fileinput.input(files=scans_tsvs, inplace=True) as f:
        for line in f:
            if not all([k in line for k in ['fmap/','_part-phase']]): # remove phase fmap
                if 'fmap/' in line:
                    line = re.sub(
                        r"^(.*/sub-.*(_ses-[^_]+))(_acq-([A-Za-z0-9]+))(.*?)(_echo-([0-9]+))(_.*)$",
                        r"\1_acq-\4Echo\7\5\8",
                        line
                    )
                print(line, end='')

    ds.save(message='fix fmap multiecho')    

def fix_complex_events(ds):
    # remove phase event files.
    new_files = [(ds.pathobj / nf) for nf in ds.repo.call_git(['show','--name-only','HEAD~1','--format=oneline']).split('\n')[1:]]
    phase_events = [f for f in ds.pathobj.glob('sub-*/ses-*/func/*_part-phase*_events.tsv') if f in new_files]
    if list(phase_events):
        ds.repo.remove(phase_events)
    # remove part-mag from remaining event files
    mag_events = [f for f in ds.pathobj.glob('sub-*/ses-*/func/*_part-mag*_events.tsv') if f in new_files]
    if len(mag_events):
        for f in mag_events:
            ds.repo.call_git(['mv', str(f), str(f).replace('_part-mag','')])
        ds.save(message='fix complex event files')
    
def main():

    args = parse_args()
    nprocs = args.nprocs

    pool = multiprocessing.Pool(nprocs)
    res = pool.map(
        partial(single_session_job,
                output_file=args.output_file,
                ria_storage_remote=args.ria_storage_remote,
                b0_field_id=args.b0_field_id,
                overwrite=args.overwrite,
        ),
        args.files)

    print("SUMMARY " + "#"*40)
    for r,f in zip(res, args.files):
        print(f"{f}: {'SUCCESS' if r is None else 'FAIL -> ' + r}")
    print("#"*50)

if __name__ == "__main__":
    main()