# git annex
source /project/def-pbellec/shared/venvs/datalad/bin/activate

datalad install --reckless ephemeral -s ria+file:///project/rrg-pbellec/ria-rorqual#~cneuromod.multfs.fmriprep@dev
cd cneuromod.multfs.fmriprep

datalad get -n --reckless ephemeral sourcedata/cneuromod.multfs.raw/

