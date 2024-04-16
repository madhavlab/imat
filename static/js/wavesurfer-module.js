
import WaveSurfer from 'https://unpkg.com/wavesurfer.js@7/dist/wavesurfer.esm.js'
import TimelinePlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/timeline.esm.js'
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'



const wavesurfer = WaveSurfer.create({
    container: "#audiowave",
    waveColor: "#808080",
    progressColor:"#0000FF" ,
    // progressColor: "#2f4f4f",
    backend: 'MediaElement',
    height: 150,
    responsive: true,
    hideScrollbar: true,
    cursorColor: "#0000FF",
    // cursorColor: "#2f4f4f",
    cursorWidth: 2,
    barwidth: 5,
    barGap: 1.5,
    skipLength: 3,
    audioRate: 1,
    plugins: [TimelinePlugin.create({
      height: 10,
      timeInterval: 0.1,
      primaryLabelInterval: 1,
      style: {
        fontSize: '10px',
        color: '#6A3274',
      },
    })]               
});

export default wavesurfer;

// wavesurfer.load("abjones_1_01.wav");

// // Play on click
// wavesurfer.on('interaction', () => {
//   wavesurfer.play()
// })

// // Rewind to the beginning on finished playing
// wavesurfer.on('finish', () => {
//   wavesurfer.setTime(0)
// })

// // Update the zoom level on slider change
// wavesurfer.once('decode', () => {
//   const slider = document.querySelector('input[type="range"]')

//   slider.addEventListener('input', (e) => {
//     const minPxPerSec = e.target.valueAsNumber
//     wavesurfer.zoom(minPxPerSec)
//   })
// })



           

