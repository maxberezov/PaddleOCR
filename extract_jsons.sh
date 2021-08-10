#!/bin/bash
images=`ls images/*`
for eachfile in $images
do
  ext='.json'

  filename=$(basename -- "$eachfile")
  filename="${filename%.*}"
  json_dir="jsons"
  json_path=$json_dir/$filename$ext
  python3 process_image.py $eachfile $json_path
done

