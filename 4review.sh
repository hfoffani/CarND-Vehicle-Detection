MUSTBE="README.md vehicle_detection.mp4 vehicle-detection.ipynb"

FILES="$MUSTBE output_images/*.png"

if [ `ls -1 $MUSTBE 2>/dev/null | wc -l` -ne 3 ]
then
    echo "missing file(s)"
    exit 1
fi
zip project5.zip $FILES

