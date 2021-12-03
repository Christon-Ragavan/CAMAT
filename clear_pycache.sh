find . | grep -E "(__pycache__|\.pyc|\music_xml_parser\ipynb\.ipynb_checkpoints|\.pyo$)" | xargs rm -rf
find . -name .DS_Store -print0 | xargs -0 git rm -f --ignore-unmatch
