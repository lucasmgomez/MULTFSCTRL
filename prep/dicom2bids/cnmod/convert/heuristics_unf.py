import os, re, glob
from frozendict import frozendict
import nibabel.nicom.dicomwrappers as nb_dw
from heudiconv.heuristics.reproin import (
    create_key,
    get_dups_marked,
    lgr,
)
from collections import OrderedDict

def load_example_dcm(seqinfo):
    ex_dcm_path = sorted(glob.glob(os.path.join('/tmp', 'heudiconv*', '*', seqinfo.dcm_dir_name, seqinfo.example_dcm_file)))[0]
    return nb_dw.wrapper_from_file(ex_dcm_path)

def custom_seqinfo(wrapper, series_files):
    image_history = ice_dims = pedir_pos = None
    if hasattr(wrapper, 'csa_header'):
        pedir_pos = wrapper.csa_header["tags"]["PhaseEncodingDirectionPositive"]["items"]
        pedir_pos = pedir_pos[0] if len(pedir_pos) else None
        image_history = ';'.join(filter(len, wrapper.csa_header['tags']['ImageHistory']['items']))
        ice_dims = wrapper.csa_header['tags']['ICE_Dims']['items'][0]

    slice_orient = wrapper.dcm_data.get([0x0051,0x100e])
    receive_coil = wrapper.dcm_data.get((0x0051,0x100f))

    custom_info = frozendict({
        'patient_name': wrapper.dcm_data.PatientName,
        'pe_dir': wrapper.dcm_data.get('InPlanePhaseEncodingDirection', None),
        'pe_dir_pos': pedir_pos,
        'body_part': wrapper.dcm_data.get("BodyPartExamined", None),
        'scan_options': str(wrapper.dcm_data.get("ScanOptions", None)),
        'image_comments': wrapper.dcm_data.get("ImageComments", ""),
        'slice_orient': str(slice_orient.value) if slice_orient else None,
        'echo_number': str(wrapper.dcm_data.get("EchoNumber", None)),
        'rescale_slope': wrapper.dcm_data.get("RescaleSlope", None),
        'receive_coil': str(receive_coil.value) if receive_coil else None,
        'image_history': image_history,
        'ice_dims': ice_dims,
    })
    return custom_info

def infotoids(seqinfos, outdir):
    seqinfo = next(iter(seqinfos))
    patient_name = str(seqinfo.custom['patient_name'])

    subject_id = 'unknown'
    session_id = 'unknown'

    m = re.search(r'(?:^|[_-])sub[_-]?([A-Za-z0-9]+)', patient_name)
    if m:
        subject_id = m.group(1)
    m = re.search(r'(?:^|[_-])ses[_-]?([A-Za-z0-9]+)', patient_name)
    if m:
        session_id = m.group(1)

    if subject_id.isdigit():
        subject_id = subject_id.zfill(2)
    if session_id.isdigit():
        session_id = session_id.zfill(2)

    return {"session": session_id, "subject": subject_id}

# --------- token helpers (task/acq/run only) ---------

def _search_tokens(sources, pattern):
    for src in sources:
        if not src:
            continue
        m = pattern.search(src)
        if m:
            return m.group(1)
    return None

def get_task(s):
    sources = [
        getattr(s, 'series_id', '') or '',
        getattr(s, 'series_description', '') or '',
        getattr(s, 'protocol_name', '') or '',
    ]
    # task-<alnum> followed by '_' or '-' or EOL
    pat = re.compile(r'(?:^|[_-])task-([A-Za-z0-9]+)(?=[_-]|$)')
    return _search_tokens(sources, pat)

def get_run(s):
    sources = [
        getattr(s, 'series_id', '') or '',
        getattr(s, 'series_description', '') or '',
        getattr(s, 'protocol_name', '') or '',
    ]
    pat = re.compile(r'(?:^|[_-])run-([0-9]+)(?=[_-]|$)')
    return _search_tokens(sources, pat)

def get_acq(s):
    sources = [
        getattr(s, 'series_id', '') or '',
        getattr(s, 'series_description', '') or '',
        getattr(s, 'protocol_name', '') or '',
    ]
    pat_acq  = re.compile(r'(?:^|[_-])acq-([A-Za-z0-9]+)(?=[_-]|$)')
    return _search_tokens(sources, pat_acq)

def pick_t1_acq(protocol_name, series_description):
    src = f"{protocol_name or ''}_{series_description or ''}"
    if re.search(r'(?:^|[_-])nd(?:$|[_-])', src, flags=re.IGNORECASE):
        return "nd"
    if re.search(r'(?:^|[_-])iso(?:$|[_-])', src, flags=re.IGNORECASE):
        return "iso"
    if re.search(r'(?:^|[_-])p2(?:$|[_-])', src, flags=re.IGNORECASE):
        return "p2"
    return None

rec_exclude = [
    "ORIGINAL",
    "PRIMARY",
    "M",
    "P",
    "MB",
    "ND",
    "MOSAIC",
    "NONE",
    "DIFFUSION",
    "UNI",
] + [f"TE{i}" for i in range(9)]

def get_seq_bids_info(s):
    seq = {"type": "anat", "label": None}
    seq_extra = {}

    pn = (s.protocol_name or "").lower()
    sd = (s.series_description or "").lower()
    sn = (s.sequence_name or "").lower()
    sid = (s.series_id or "")

    # ImageType handling
    itype = [str(x) for x in getattr(s, "image_type", [])]
    for it in itype[2:]:
        if it not in rec_exclude:
            seq_extra["rec"] = it.lower()
    seq_extra["part"] = "mag" if "M" in itype else ("phase" if "P" in itype else None)

    # Phase-encoding direction (best effort)
    try:
        pedir = s.custom['pe_dir']
        pedir = "AP" if "COL" in pedir else "LR"
        pedir_pos = bool(s.custom['pe_dir_pos'])
        seq["dir"] = pedir if pedir_pos else pedir[::-1]
    except Exception:
        pass

    # label non-brain if present
    bodypart = s.custom['body_part']
    if bodypart and bodypart != "BRAIN":
        seq["bp"] = bodypart.lower()

    image_comments = (s.custom['image_comments'] or "").lower()
    is_sbref = "single-band reference" in image_comments

    if s.custom['ice_dims'] and s.custom['ice_dims'][0] != 'X':
        seq['rec'] = 'uncombined'

    # ---------- ANATS ----------
    if "localizer" in pn or "localizer" in sd:
        seq["label"] = "localizer"
        slice_orient = s.custom['slice_orient']
        if slice_orient:
            seq_extra['acq'] = slice_orient.lower()

    elif "aahead" in pn or "scout" in pn or "aahead" in sd or "scout" in sd:
        seq["label"] = "scout"

    elif "mpr_" in sd:
        m = re.search(r"mpr_(cor|sag|tra)", sd)
        if m:
            seq["label"] = "localizer"
            seq_extra["acq"] = m.group(1)

    elif (s.dim4 == 1) and (
        re.search(r'\b(mprage|t1)\b', pn) or
        re.search(r'\b(mprage|t1)\b', sd) or
        'tfl' in sn or 'tfl3d' in sn
    ):
        seq["label"] = "T1w"
        acq_t1 = pick_t1_acq(s.protocol_name, s.series_description)
        if acq_t1:
            seq["acq"] = acq_t1

    elif (s.dim4 == 1) and ("t2" in pn) and ("spc_314ns" in sn):
        seq["label"] = "T2w"

    elif ("*tfl3d1_16" in sn) and (s.dim4 == 1) and ("mp2rage" in pn) and ("memp2rage" not in pn):
        seq["label"] = "MP2RAGE"
        if "inv1" in sd:
            seq["inv"] = 1
        elif "inv2" in sd:
            seq["inv"] = 2
        elif "UNI" in itype:
            seq["label"] = "UNIT1"

    elif "*fl3d1" in sn and ("scout" not in sd and "localizer" not in sd):
        seq["label"] = "MTS"
        seq["mt"] = "on" if s.custom['scan_options'] == "MT" else "off"
        seq["flip"] = 2 if 't1w' in sid.lower() else 1

    # fmap types
    elif "tfl2d1" in sn:
        seq["type"] = "fmap"
        seq["label"] = "TB1TFL"
        seq["acq"] = "famp" if "flip angle map" in image_comments else "anat"

    elif "fm2d2r" in sn:
        seq["type"] = "fmap"
        seq["label"] = "phasediff" if "phase" in itype else f"magnitude{s.custom['echo_number']}"

    elif (s.dim4 == 1) and ("swi3d1r" in sn):
        seq["type"] = "swi"
        seq["label"] = "swi" if "MNIP" not in itype else "minIP"

    elif (("ep_b" in sn) or ("ez_b" in sn) or ("epse2d1_110" in sn)) and not any(t in itype for t in ["DERIVED", "PHYSIO"]):
        seq["type"] = "dwi"
        seq["label"] = "sbref" if is_sbref else "dwi"
        seq_extra["part"] = 'phase' if s.custom['rescale_slope'] else 'mag'

    # ---------- FUNC ----------
    elif "epfid2d" in sn:
        seq["task"] = get_task(s)
        if "AP" in s.series_id and not seq["task"]:
            seq["type"] = "fmap"
            seq["label"] = "epi"
            seq["acq"] = "sbref" if is_sbref else "bold"
        else:
            seq["type"] = "func"
            seq["label"] = "sbref" if is_sbref else "bold"

        acq = get_acq(s)
        if acq:
            seq["acq"] = acq

        seq["run"] = get_run(s)
        if s.is_motion_corrected:
            seq["rec"] = "moco"

    # sbref shouldn’t carry part label
    if seq["label"] == "sbref" and "part" in seq_extra:
        del seq_extra["part"]

    return seq, seq_extra

def generate_bids_key(seq_type, seq_label, prefix, bids_info, show_dir=False, outtype=("nii.gz",), **bids_extra):
    bids_info.update(bids_extra)
    suffix_parts = [
        None if not bids_info.get("task") else "task-%s" % bids_info["task"],
        None if not bids_info.get("acq")  else "acq-%s"  % bids_info["acq"],
        None if not bids_info.get("ce")   else "ce-%s"   % bids_info["ce"],
        None if not (bids_info.get("dir") and show_dir) else "dir-%s" % bids_info["dir"],
        None if not bids_info.get("rec")  else "rec-%s"  % bids_info["rec"],
        None if not bids_info.get("inv")  else "inv-%d"  % int(bids_info["inv"]),
        None if not bids_info.get("tsl")  else "tsl-%d"  % int(bids_info["tsl"]),
        None if not bids_info.get("loc")  else "loc-%s"  % bids_info["loc"],
        None if not bids_info.get("bp")   else "bp-%s"   % bids_info["bp"],
        None if not bids_info.get("run")  else "run-%02d"% int(bids_info["run"]),
        None if not bids_info.get("echo") else "echo-%d" % int(bids_info["echo"]),
        None if not bids_info.get("flip") else "flip-%d" % int(bids_info["flip"]),
        None if not bids_info.get("mt")   else "mt-%s"   % bids_info["mt"],
        None if not bids_info.get("part") else "part-%s" % bids_info["part"],
        seq_label,
    ]
    suffix = "_".join(filter(bool, suffix_parts))
    return create_key(seq_type, suffix, prefix=prefix, outtype=outtype)

def infotodict(seqinfo):
    info = OrderedDict()
    skipped, skipped_unknown = [], []
    current_run = 0
    run_label = None
    dcm_image_iod_spec = None
    skip_derived = True

    outtype = ("nii.gz",)
    sbref_as_fieldmap = False   # leave False unless you want sbrefs duplicated into fmap
    prefix = ""

    fieldmap_runs = {}
    all_bids_infos = {}

    for s in seqinfo:
        bids_info, bids_extra = get_seq_bids_info(s)
        all_bids_infos[s.series_id] = (bids_info, bids_extra)

        # Debug: what each func resolves to (now task/acq only)
        if bids_info.get("type") == "func" and bids_info.get("label") == "bold":
            lgr.info(
                f"[FUNC DEBUG] series_id={s.series_id} "
                f"series_desc={getattr(s, 'series_description', '')} "
                f"task={bids_info.get('task')} "
                f"acq={bids_info.get('acq')} "
                f"run={bids_info.get('run')}"
            )

        if (
            skip_derived
            and (s.is_derived or ("DERIVED" in s.image_type) or ("PHYSIO" in s.image_type))
            and not s.is_motion_corrected
            and "UNI" not in s.image_type
        ):
            skipped.append(s.series_id)
            lgr.debug("Ignoring derived data %s", s.series_id)
            continue

        seq_type = bids_info["type"]
        seq_label = bids_info["label"]

        # sbref→fmap duplication path (no desc used)
        if ((seq_type == "fmap" and seq_label == "epi") or
            (sbref_as_fieldmap and seq_label == "sbref" and seq_type=='func')
        ) and bids_info.get("part") in ["mag", None]:
            pe_dir = bids_info.get("dir", None)
            if pe_dir not in fieldmap_runs:
                fieldmap_runs[pe_dir] = 0
            fieldmap_runs[pe_dir] += 1
            run_id = fieldmap_runs[pe_dir]

            if sbref_as_fieldmap and seq_label == "sbref":
                suffix_parts = [
                    "acq-sbref",
                    None if not bids_info.get("ce") else f"ce-{bids_info['ce']}",
                    None if not pe_dir else f"dir-{bids_info['dir']}",
                    f"run-{run_id:02d}",
                    "epi",
                ]
                suffix = "_".join(filter(bool, suffix_parts))
                template = create_key("fmap", suffix, prefix=prefix, outtype=outtype)
                info.setdefault(template, []).append(s.series_id)

        show_dir = seq_type in ["fmap", "dwi"] and not seq_label=='TB1TFL'
        template = generate_bids_key(seq_type, seq_label, prefix, bids_info, show_dir, outtype)
        info.setdefault(template, []).append(s.series_id)

    if skipped:
        lgr.info("Skipped %d sequences: %s" % (len(skipped), skipped))
    if skipped_unknown:
        lgr.warning(
            "Could not figure out where to stick %d sequences: %s"
            % (len(skipped_unknown), skipped_unknown)
        )

    info = dedup_bids_extra(info, all_bids_infos)
    info = get_dups_marked(info)
    info = dict(info)

    for k, i in info.items():
        lgr.info(f"{k} {i}")

    return info

def dedup_bids_extra(info, bids_infos):
    # add `rec-` or `part-` to dedup series originating from the same acquisition
    info = info.copy()
    for template, series_ids in list(info.items()):
        if len(series_ids) >= 2:
            lgr.warning("Detected %d run(s) for template %s: %s",
                        len(series_ids), template[0], series_ids)

            # No desc-based dedup anymore
            for extra in ["rec", "part"]:
                bids_extra_values = [bids_infos[sid][1].get(extra) for sid in series_ids]
                lgr.info(f'{extra} values {bids_extra_values}')
                if len(set(bids_extra_values)) < 2:
                    continue

                lgr.info(f"dedup series using {extra}")

                for sid in list(series_ids):
                    series_bids_info, series_bids_extra = bids_infos[sid]
                    new_template = generate_bids_key(
                        series_bids_info["type"],
                        series_bids_info["label"],
                        "",
                        series_bids_info,
                        show_dir=series_bids_info["type"] in ["fmap", "dwi"],
                        outtype=("nii.gz",),
                        **{extra: series_bids_extra.get(extra)}
                    )
                    info.setdefault(new_template, []).append(sid)
                    info[template].remove(sid)
                    if not info[template]:
                        del info[template]
                break
    return info