// Handeling Furigana

document.getElementById('noFuri').addEventListener('click', function() {
    changeContent('data-no-furi');
});

document.getElementById('allFuri').addEventListener('click', function() {
    changeContent('data-all-furi');
});

document.getElementById('firstFuri').addEventListener('click', function() {
    changeContent('data-first-furi');
});

function changeContent(attribute) {
    const paragraphs = document.querySelectorAll('#japaneseContent p');

    paragraphs.forEach(paragraph => {
        paragraph.innerHTML = paragraph.getAttribute(attribute);
    });
}


// Handeling the audio
const globalPlayer = document.querySelector('.middle-bottom .audio-widget');
const playBtn = globalPlayer.querySelector('.play-btn');
const pauseBtn = globalPlayer.querySelector('.pause-btn');
const stopBtn = globalPlayer.querySelector('.stop-btn');
const progressBar = globalPlayer.querySelector('.progress');
let audio = null;  // We'll store our Audio instance here

document.querySelectorAll('.play-btn').forEach(button => {
    button.addEventListener('click', function() {
        if(audio) { // If there's already an audio instance, stop and remove it
            audio.pause();
            audio = null;
        }

        const audioFile = button.getAttribute('data-audio');
        audio = new Audio(audioFile);
        audio.play();
        playBtn.hidden = true;
        pauseBtn.hidden = false;

        audio.addEventListener('timeupdate', function() {
            const progressPercentage = (audio.currentTime / audio.duration) * 100;
            progressBar.style.width = progressPercentage + '%';

            if (audio.currentTime === audio.duration) {
                playBtn.hidden = false;
                pauseBtn.hidden = true;
            }
        });
    });
});

pauseBtn.addEventListener('click', function() {
    if(audio) {
        audio.pause();
        playBtn.hidden = false;
        pauseBtn.hidden = true;
    }
});

stopBtn.addEventListener('click', function() {
    if(audio) {
        audio.pause();
        audio.currentTime = 0;
        playBtn.hidden = false;
        pauseBtn.hidden = true;
    }
});

audio.addEventListener('timeupdate', function() {
    const progressPercentage = (audio.currentTime / audio.duration) * 100;
    progressBar.style.width = progressPercentage + '%';

    if (audio.currentTime === audio.duration) {
        playBtn.hidden = false;
        pauseBtn.hidden = true;
    }
});