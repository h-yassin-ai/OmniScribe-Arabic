process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

const { facebook } = require('@mrnima/facebook-downloader');
const fs = require('fs');
const path = require('path');
const axios = require('axios');

// Usage: node fb_downloader.js <url> <output_dir>
const url = process.argv[2];
const outputDir = process.argv[3];

if (!url || !outputDir) {
    console.error(JSON.stringify({ error: "Missing arguments: url or output_dir" }));
    process.exit(1);
}

if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

async function downloadVideo() {
    try {
        const result = await facebook(url);

        if (!result || !result.status) {
            throw new Error("Failed to fetch video metadata");
        }

        const data = result.data;
        // Prioritize HD, then SD
        const videoUrl = data.hd || data.sd;

        if (!videoUrl) {
            throw new Error("No video URL found in response");
        }

        const title = data.title || `facebook_video_${Date.now()}`;
        // Sanitize filename
        const safeTitle = title.replace(/[^a-z0-9]/gi, '_').substring(0, 50);
        const filename = `${safeTitle}.mp4`;
        const filePath = path.join(outputDir, filename);

        // Check if file already exists
        if (fs.existsSync(filePath)) {
            console.log(JSON.stringify({
                status: "success",
                title: title,
                path: filePath,
                id: data.id || "unknown",
                already_exists: true
            }));
            return;
        }

        // Download the file using axios
        const writer = fs.createWriteStream(filePath);
        const response = await axios({
            url: videoUrl,
            method: 'GET',
            responseType: 'stream'
        });

        response.data.pipe(writer);

        await new Promise((resolve, reject) => {
            writer.on('finish', resolve);
            writer.on('error', reject);
        });

        console.log(JSON.stringify({
            status: "success",
            title: title,
            path: filePath,
            id: data.id || "unknown"
        }));

    } catch (error) {
        console.error(JSON.stringify({ error: error.message }));
        process.exit(1);
    }
}

downloadVideo();
