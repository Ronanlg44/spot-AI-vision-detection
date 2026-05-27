#!/usr/bin/env bash
# Download Spot footage used to train the YOLOv8m detector.
#
# Two sources:
#   - DARPA Subterranean Challenge Final Event (Day 2, 3, 4) -- specific
#     passages, downloaded with timestamps.
#   - Boston Dynamics promotional videos (Spot Launch, Spot Warehouse) --
#     downloaded in full.
#
# Requires yt-dlp: https://github.com/yt-dlp/yt-dlp

set -euo pipefail

TARGET_DIR="data/videos"
mkdir -p "${TARGET_DIR}"
cd "${TARGET_DIR}"

URL_DAY2="https://www.youtube.com/watch?v=8Y_6p_jhmCA"
URL_DAY3="https://www.youtube.com/watch?v=jNb6vf89q-M"
URL_DAY4="https://www.youtube.com/watch?v=EAPSm7udG3Q"
URL_SPOT_LAUNCH="https://www.youtube.com/watch?v=wlkCQXHEgjA"
URL_SPOT_WAREHOUSE="https://www.youtube.com/watch?v=YD9EaS3VRbc"

download_section() {
    local url="$1"
    local section="$2"
    local output="$3"

    echo "  -> ${output}"
    yt-dlp -f mp4 --download-sections "*${section}" "${url}" -o "${output}"
}

download_full() {
    local url="$1"
    local output="$2"

    echo "  -> ${output}"
    yt-dlp -f mp4 "${url}" -o "${output}"
}

echo "DARPA Day 2 passages..."
download_section "${URL_DAY2}" "02:23:01-02:23:58" "day2_passage1.mp4"
download_section "${URL_DAY2}" "02:36:42-02:37:06" "day2_passage2.mp4"
download_section "${URL_DAY2}" "02:38:16-02:38:36" "day2_passage3.mp4"
download_section "${URL_DAY2}" "02:08:54-02:09:20" "day2_passage4.mp4"
download_section "${URL_DAY2}" "01:59:01-02:01:50" "day2_passage5_fake.mp4"

echo "DARPA Day 3 passages..."
download_section "${URL_DAY3}" "01:27:26-01:29:02" "day3_passage1.mp4"
download_section "${URL_DAY3}" "01:50:22-01:53:58" "day3_passage2.mp4"

echo "DARPA Day 4 passages..."
download_section "${URL_DAY4}" "01:41:51-01:42:37" "day4_passage1.mp4"
download_section "${URL_DAY4}" "01:46:16-01:47:12" "day4_passage2.mp4"
download_section "${URL_DAY4}" "01:54:54-01:56:32" "day4_passage3.mp4"
download_section "${URL_DAY4}" "01:56:35-01:57:32" "day4_passage4_fake.mp4"
download_section "${URL_DAY4}" "02:00:16-02:00:52" "day4_passage5.mp4"
download_section "${URL_DAY4}" "02:11:16-02:12:03" "day4_passage6.mp4"
download_section "${URL_DAY4}" "02:13:55-02:14:38" "day4_passage7.mp4"
download_section "${URL_DAY4}" "02:16:33-02:16:57" "day4_passage8.mp4"

echo "Boston Dynamics promotional clips..."
download_full "${URL_SPOT_LAUNCH}" "spot_launch.mp4"
download_full "${URL_SPOT_WAREHOUSE}" "spot_warehouse.mp4"

# Two additional negative passages (empty_spot1, empty_spot2) were used in
# the original training run. Their exact timestamps were not preserved.
# The pipeline remains functional without them: day2_passage5_fake.mp4 and
# day4_passage4_fake.mp4 already provide negative samples.

echo "Done."
