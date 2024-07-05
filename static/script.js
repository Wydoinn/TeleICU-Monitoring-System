const videoFeed = document.getElementById('video-feed');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');

let isRunning = false;

startButton.addEventListener('click', startVideo);
stopButton.addEventListener('click', stopVideo);

function startVideo() {
    if (!isRunning) {
        isRunning = true;
        fetch('/start')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'started') {
                    videoFeed.src = '/video_feed';
                    videoFeed.style.opacity = 0;
                    setTimeout(() => {
                        videoFeed.style.opacity = 1;
                    }, 100);
                    startButton.disabled = true;
                    stopButton.disabled = false;
                }
            });
    }
}

function stopVideo() {
    if (isRunning) {
        isRunning = false;
        fetch('/stop')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'stopped') {
                    videoFeed.src = '';
                    videoFeed.style.backgroundColor = 'white';
                    videoFeed.style.opacity = 0;
                    startButton.disabled = false;
                    stopButton.disabled = true;
                }
            });
    }
}