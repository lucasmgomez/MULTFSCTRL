docker run --rm -it \
  -v "$PWD:/work" \
  --user 0:0 \
  alpine:latest \
  sh -lc 'rm -rf /mnt/tempdata/lucas/fmri/recordings/TR/neural/fmriprep_outs/first_run/bids/sub-01'