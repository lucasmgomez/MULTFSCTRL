# python
import json, sys
import nibabel as nib
import numpy as np
from pathlib import Path

nii = Path(sys.argv[1])
jfile = nii.with_suffix('.json') if nii.suffix == '.nii' else nii.with_suffix('').with_suffix('.json')
img = nib.load(str(nii))
hdr = img.header
data = img.get_fdata(dtype=np.float32)

print("file:", nii)
print("shape:", img.shape)
print("dtype:", data.dtype)
print("voxel sizes (zooms):", hdr.get_zooms())
print("affine:\n", img.affine)
print("nonzero voxels:", np.count_nonzero(data))
if jfile.exists():
    j = json.loads(jfile.read_text())
    for k in ("RepetitionTime","EchoTime","EffectiveEchoSpacing","EchoNumber","SpacingBetweenSlices"):
        if k in j: print(k, "=", j[k])
else:
    print("JSON sidecar missing:", jfile)