"""
movie: A replacement for scitools.std.movie.

Creates HTML animations from sequences of image files.
"""

import glob
import os


def movie(
    input_pattern,
    encoder="html",
    fps=4,
    output_file="movie.html",
    **kwargs,
):
    """
    Create a movie/animation from a sequence of image files.

    Parameters
    ----------
    input_pattern : str
        Glob pattern for input files, e.g., 'tmp_*.png'
    encoder : str, optional
        Output format: 'html' for HTML5 animation (default)
    fps : int, optional
        Frames per second (default: 4)
    output_file : str, optional
        Output filename (default: 'movie.html')
    **kwargs : dict
        Additional arguments (ignored, for compatibility)

    Returns
    -------
    str
        Path to the created movie file

    Examples
    --------
    >>> movie('frame_*.png', encoder='html', fps=10, output_file='animation.html')
    """
    # Get sorted list of input files
    files = sorted(glob.glob(input_pattern))

    if not files:
        raise ValueError(f"No files found matching pattern: {input_pattern}")

    if encoder == "html":
        return _make_html_movie(files, fps, output_file)
    else:
        raise ValueError(
            f"Encoder '{encoder}' not supported. Use 'html' or "
            "use ffmpeg directly via subprocess."
        )


def _make_html_movie(files, fps, output_file):
    """Create an HTML5 animation with JavaScript controls."""
    import base64

    # Read and encode all images
    images_data = []
    for filepath in files:
        with open(filepath, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(filepath)[1].lower()
            mime = "image/png" if ext == ".png" else "image/jpeg"
            images_data.append(f"data:{mime};base64,{data}")

    interval = int(1000 / fps)  # milliseconds between frames

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Animation</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            text-align: center;
            padding: 20px;
            background: #f0f0f0;
        }}
        #container {{
            display: inline-block;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        #animation {{
            max-width: 100%;
            border: 1px solid #ddd;
        }}
        .controls {{
            margin-top: 15px;
        }}
        button {{
            padding: 8px 16px;
            margin: 0 5px;
            font-size: 14px;
            cursor: pointer;
            border: none;
            border-radius: 4px;
            background: #4CAF50;
            color: white;
        }}
        button:hover {{
            background: #45a049;
        }}
        button:disabled {{
            background: #cccccc;
            cursor: not-allowed;
        }}
        #frameInfo {{
            margin-top: 10px;
            color: #666;
        }}
        input[type="range"] {{
            width: 200px;
            margin: 10px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <img id="animation" src="" alt="Animation frame">
        <div class="controls">
            <button id="playPause">Pause</button>
            <button id="prevFrame">Previous</button>
            <button id="nextFrame">Next</button>
            <button id="restart">Restart</button>
        </div>
        <div>
            <input type="range" id="frameSlider" min="0" max="{len(images_data) - 1}" value="0">
        </div>
        <div id="frameInfo">Frame: 1 / {len(images_data)}</div>
        <div>
            <label>Speed: </label>
            <input type="range" id="speedSlider" min="1" max="30" value="{fps}">
            <span id="fpsDisplay">{fps} fps</span>
        </div>
    </div>

    <script>
        const images = {images_data};
        let currentFrame = 0;
        let playing = true;
        let interval = {interval};
        let timer;

        const img = document.getElementById('animation');
        const playPauseBtn = document.getElementById('playPause');
        const frameSlider = document.getElementById('frameSlider');
        const frameInfo = document.getElementById('frameInfo');
        const speedSlider = document.getElementById('speedSlider');
        const fpsDisplay = document.getElementById('fpsDisplay');

        function showFrame(n) {{
            currentFrame = n;
            if (currentFrame >= images.length) currentFrame = 0;
            if (currentFrame < 0) currentFrame = images.length - 1;
            img.src = images[currentFrame];
            frameSlider.value = currentFrame;
            frameInfo.textContent = `Frame: ${{currentFrame + 1}} / ${{images.length}}`;
        }}

        function nextFrame() {{
            showFrame(currentFrame + 1);
        }}

        function startAnimation() {{
            if (timer) clearInterval(timer);
            timer = setInterval(nextFrame, interval);
        }}

        function togglePlay() {{
            playing = !playing;
            playPauseBtn.textContent = playing ? 'Pause' : 'Play';
            if (playing) {{
                startAnimation();
            }} else {{
                clearInterval(timer);
            }}
        }}

        playPauseBtn.addEventListener('click', togglePlay);

        document.getElementById('prevFrame').addEventListener('click', () => {{
            showFrame(currentFrame - 1);
        }});

        document.getElementById('nextFrame').addEventListener('click', () => {{
            showFrame(currentFrame + 1);
        }});

        document.getElementById('restart').addEventListener('click', () => {{
            showFrame(0);
            if (!playing) togglePlay();
        }});

        frameSlider.addEventListener('input', (e) => {{
            showFrame(parseInt(e.target.value));
        }});

        speedSlider.addEventListener('input', (e) => {{
            const fps = parseInt(e.target.value);
            interval = Math.floor(1000 / fps);
            fpsDisplay.textContent = `${{fps}} fps`;
            if (playing) startAnimation();
        }});

        // Initialize
        showFrame(0);
        startAnimation();
    </script>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html)

    print(f"Created HTML animation: {output_file} ({len(files)} frames at {fps} fps)")
    return output_file
