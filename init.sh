% % bash
# If your project has a 'requirements.txt' file, we'll install it here apart from blacklisted packages that interfere with Deepnote (see above).
if test - f requirements.txt
   then
        sed - i '/jedi/d;/jupyter/d;' ./requirements.txt
        pip install - r ./requirements.txt
    else echo "No requirements.txt file."
fi
