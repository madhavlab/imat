import wavesurfer from '../js/wavesurfer-module.js';
import RegionsPlugin from 'https://unpkg.com/wavesurfer.js@7/dist/plugins/regions.esm.js'

// Global variables
const wsRegions = wavesurfer.registerPlugin(RegionsPlugin.create())
let preservePitch = true
let selectedFile = null;
let activeRegion = null;
const speeds = [0.25, 0.5, 1, 2, 4];
var spinner = document.getElementById("id_spinner_container");

var retrainPoints = {}
var selectedPoints = {}

var trace = null;
var melody_trace = null;
var newRange = null;
var chartData
const helpers = Chart.helpers
var canvasData
let isDragging = false;
var activePoint
var boxVisible = false;
var myChart
var canvas
var ctx
var selectionStart = { x: 0, y: 0 };
var selectionEnd = { x: 0, y: 0 };
var xValuesRange = null;
var init_index
var index
var yValue = null
var xValuesRangeLength = null;
var totalConfValues = null;
var cValues = null;


// ####################################################################################
// Add a Spinner

function showSpinner(spinner_text) {
    // show spinner
    const spinnerTextElement = document.getElementById("id_spinner_text");
    // Set text content of span element
    if(spinner_text==null || spinner_text==undefined){
      spinner_text = "Loading..."
    }
    spinnerTextElement.textContent = spinner_text;

    spinner.style.display = 'block';
}

function hideSpinner() {
    spinner.style.display = 'none';
}
hideSpinner()

// ####################################################################################


var onChangeVolume = function (e) {
    wavesurfer.setVolume(e.target.value); 
    };


function getConfValues() {
    let formData = new FormData();
    formData.append("file", selectedFile);
    
    fetch("/get_conf_values", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        totalConfValues = data
        var startIndx = Math.ceil((newRange[0]).toFixed(2)*100)
        cValues = data.slice(startIndx,startIndx+xValuesRangeLength)
        // console.log('original conf values:',cValues)
        generateGradientPlot(cValues)
        hideSpinner()
    })
}

function generateGradientColors(values) {
    var colors = [];
    var gradientSteps = values.length - 1;
    var colorStep = 1 / gradientSteps;
  
    for (var i = 0; i < gradientSteps; i++) {
      var color = 'rgba(0, 153, 38, ' + values[i] + ')';
      colors.push(color + ' ' + (i * colorStep * 100) + '%');
      colors.push(color + ' ' + ((i + 1) * colorStep * 100) + '%');
    }
  
    return colors;
  }

// function generateGradientColors(values) {
//     var colors = values.map(alpha => `rgba(0, 153, 38, ${alpha})`)
//     return colors
// }

function generateGradientPlot(conf_frame_val){
    var gradientColors = generateGradientColors(conf_frame_val);
    var gradientBar = document.getElementById('js-confidence-bar');
    // gradientBar.style.background = 'none';
    gradientBar.style.background = 'linear-gradient(to right, ' + gradientColors.join(', ') + ')';
}

function renderSelectionBox() {
    const x = Math.min(selectionStart.x, selectionEnd.x);
    const y = Math.min(selectionStart.y, selectionEnd.y);
    const width = Math.abs(selectionEnd.x - selectionStart.x);
    const height = Math.abs(selectionEnd.y - selectionStart.y);

    ctx.fillStyle = 'rgba(0, 0, 255, 0.3)';
    ctx.fillStyle = 'blue';
    ctx.fillRect(x, y, width, height);
}

function multiple_update_melodytrace() {
    console.log("updating multi");
  
    // retrainPoints.push([index,updatedYvalue])
  
    const layout = {
      // xaxis: {
      //   range: newRange,
      // },
      yaxis: {
        title: "Frequency (Hz)",
        fixedrange: true,
        range: [0, 4000],
        autorange: false,
      },
      margin: {
        b: 20,
      },
    };
    const config = { responsive: true };
    Plotly.update(
      "js-display-spectrogram",
      [trace, melody_trace],
      layout,
      config
    );
  }

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function handleRightClick(event) {
    event.preventDefault(); // Prevent the context menu from appearing
    // clearCanvas();
    boxVisible = false;
    // myChart.update();
}

function adjustSelectedPoints() {
    var temp = getPointsInsideSelectionBox();
    selectedPoints['x'] = [];
    selectedPoints['y'] = [];
    // Group the data based on the key values
    temp.forEach(item => {
      selectedPoints['x'].push(Math.round(item['x']*100));
      selectedPoints['y'].push(item['y']);
    });
}

function getPointsInsideSelectionBox() {
    const xMin = Math.min(selectionStart.x, selectionEnd.x);
    const xMax = Math.max(selectionStart.x, selectionEnd.x);
    const yMin = Math.min(selectionStart.y, selectionEnd.y);
    const yMax = Math.max(selectionStart.y, selectionEnd.y);

    myChart.update();

    return canvasData.filter(point => {
        const x = myChart.scales['x-axis-0'].getPixelForValue(point.x);
        const y = myChart.scales['y-axis-0'].getPixelForValue(point.y);
        return x >= xMin && x <= xMax && y >= yMin && y <= yMax;
    });
}


// function removeButtonClickHandler() {
//     showSpinner("")
//     if(selectedPoints!={} && selectedPoints['x']!=undefined){
//         init_index = Math.ceil((newRange[0]).toFixed(2)*100)   //parseFloat((newRange[0]).toFixed(2))*100 

//         var yIdx = 0
//         selectedPoints['x'].forEach(x => {
//             myChart.data.datasets[0].data[x-init_index] = 0
//             // console.log('values:',x,selectedPoints['y'][yIdx],init_index)
//             selectedPoints['y'][yIdx] = 0 //myChart.data.datasets[0].data[x-init_index];
//             retrainPoints.push([x,0])
//             // console.log('selected points',selectedPoints['y'][yIdx])
//             yIdx++
//         })            
//     }
//     console.log('remove points retrain...',retrainPoints)

//     myChart.update()

//     if(selectedPoints!={} && selectedPoints['x']!=undefined){
//         selectedPoints['x'].forEach(x=>{
//             update_melodytrace(x,0)
//         })
//     }
//     hideSpinner();
// }


function removeButtonClickHandler() {
    showSpinner("")
    if(selectedPoints!={} && selectedPoints['x']!=undefined){
        init_index = Math.ceil((newRange[0]).toFixed(2)*100)   //parseFloat((newRange[0]).toFixed(2))*100 

        var yIdx = 0
        selectedPoints['x'].forEach(x => {
            myChart.data.datasets[0].data[x-init_index] = 0
            // console.log('values:',x,selectedPoints['y'][yIdx],init_index)
            selectedPoints['y'][yIdx] = 0 //myChart.data.datasets[0].data[x-init_index];
            retrainPoints.push([x,0])
            // console.log('selected points',selectedPoints['y'][yIdx])
            yIdx++
        })            
    }
    console.log('remove points retrain...',retrainPoints)

    myChart.update()

    if(selectedPoints!={} && selectedPoints['x']!=undefined){
        selectedPoints['x'].forEach(x=>{
            melody_trace.y[x] = 0;
        })
    }
    multiple_update_melodytrace();
    hideSpinner();
}


function dragRect() {
    canvas.addEventListener('mousedown',(event)=>{
        // console.log('drag rect down newRange',newRange)

        if (boxVisible == false) {
            isDragging = true;
            boxVisible = true;
            myChart.options.events = false;
            selectionStart = helpers.getRelativePosition(event, myChart)
            selectionEnd = {...selectionStart}
            // console.log('selection start',selectionStart)
        }
    })

    canvas.addEventListener('contextmenu',handleRightClick);

    canvas.addEventListener('mousemove',(event)=>{
        if (isDragging) {
            selectionEnd = helpers.getRelativePosition(event, myChart)
            // console.log('drag rect move newRange',newRange)

            // console.log('selection end:',selectionEnd)
            renderSelectionBox();
        }
    })

    canvas.addEventListener('mouseup',(event)=> {
        if (isDragging) {
            isDragging = false;
            myChart.options.events = true;
            adjustSelectedPoints();

            const removeButton = document.querySelector('.js-remove-btn');

            removeButton.removeEventListener("click", removeButtonClickHandler);

            removeButton.addEventListener("click", removeButtonClickHandler); 
            
        }
    })
}


function updateChart(xValuesRange,filteredYValues) {
    console.log('chart updating...')
    chartData = {
        labels: xValuesRange,
        datasets: [{
            data: filteredYValues,
            label: 'Annotation',
            borderColor: 'red',
            borderWidth: 2,
            pointRadius: 2,
            fill: false
            }]
    }


    canvasData = xValuesRange.map((x,index)=>({x:x,y:filteredYValues[index]}))

    if (myChart) {
        myChart.data.labels = chartData.labels;
        myChart.data.datasets[0].data = chartData.datasets[0].data;
        myChart.update();
    } else {
        canvas = document.getElementById('overlay-canvas');
        ctx = canvas.getContext("2d");
        myChart = new Chart (ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                scales: {
                    xAxes: [{
                    display: true,
                    
                    }], 
                    
                    yAxes: [{
                    display: true,
                    ticks: {
                        beginAtZero: true,
                        max: 4000,
                        min: 0
                    }
                    }]
                },
                tooltips: {
                    enabled: false  // Set to false to disable tooltips
                }
            }
        })
    }   

    canvas.onpointerdown = down_handler;
    canvas.onpointerup = up_handler;
    canvas.onpointermove = null;

    dragRect();
}

function down_handler(event) {
    const points = myChart.getElementAtEvent(event,{intersect:false})
    // console.log('down_handler newRange',newRange)
    // console.log('entered down_handler')
    if (points.length>0) {
        activePoint = points[0]
        canvas.onpointermove = move_handler
    } 

}

function map(value, start1, stop1, start2, stop2) {
    return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1))
};

function move_handler(event) {
    init_index = parseFloat((newRange[0]).toFixed(2))*100 

    if (activePoint != null) {
        var data = activePoint._chart.data
        
        index = init_index +activePoint['_index']    
        var datasetIndex = activePoint._datasetIndex

        //read mouse position
        var position = helpers.getRelativePosition(event, myChart)

        // convert mouse position to chart y axis value
        var chartArea = myChart.chartArea
        var yAxis = myChart.scales["y-axis-0"];
        yValue = map(position.y, chartArea.bottom, chartArea.top, yAxis.min, yAxis.max);

        if( yValue<0 ){
            data.datasets[datasetIndex].data[activePoint._index] = 0;
        }
        else{
            if(selectedPoints!={} && selectedPoints['x']!=undefined && selectedPoints['x'].includes(activePoint._index+init_index)){
                var diff = yValue-data.datasets[datasetIndex].data[activePoint._index];
                var yIdx = 0;
                selectedPoints['x'].forEach(x =>{
                    data.datasets[datasetIndex].data[x-init_index]+=diff;
                    if(data.datasets[datasetIndex].data[x-init_index]<0){
                        data.datasets[datasetIndex].data[x-init_index] = 0;
                    }
                    selectedPoints['y'][yIdx] = data.datasets[datasetIndex].data[x-init_index];
                    yIdx++;
                });
            }
            else{
                data.datasets[datasetIndex].data[activePoint._index] = yValue;
            }
        }
        myChart.update(); 
       
    }
}

function removeDuplicateEntries(arr) {
    // Use a Set to keep track of unique entries
    const uniqueEntries = new Set();

    // Filter out duplicates and only keep unique entries
    const uniqueArray = arr.filter(entry => {
        const entryString = JSON.stringify(entry);
        if (!uniqueEntries.has(entryString)) {
            uniqueEntries.add(entryString);
            return true;
        }
        return false;
    });

    return uniqueArray;
}

// function up_handler(event) {
//     if(activePoint!=null) {
//         canvas.onpointermove = null; 
//         activePoint = null;
//         // console.log('updated index and yvalue',index,yValue)
//         if (yValue<0) {
//             retrainPoints.push([index,0])
//             update_melodytrace(index,0);
//         } else{ 
//             retrainPoints.push([index,yValue])
//             update_melodytrace(index,yValue);
//         }
//         if(selectedPoints!={} && selectedPoints['x']!=undefined && selectedPoints['x'].includes(index)){
//             var yIdx = 0
//             selectedPoints['x'].forEach(x => {
//                 // console.log(x,melody_trace.y[x],selectedPoints['y'][0])
//                 retrainPoints.push([x,selectedPoints['y'][yIdx]])
//                 update_melodytrace(x,selectedPoints['y'][yIdx])

//                 yIdx++
//             })

//         }
//         retrainPoints = removeDuplicateEntries(retrainPoints)
//         console.log('retrain points',retrainPoints)
//     }

// }

function up_handler(event) {
    if(activePoint!=null) {
        canvas.onpointermove = null; 
        activePoint = null;
        // console.log('updated index and yvalue',index,yValue)
        if (yValue<0) {
            retrainPoints.push([index,0])
            update_melodytrace(index,0);
        } else{ 
            retrainPoints.push([index,yValue])
            update_melodytrace(index,yValue);
        }
        if(selectedPoints!={} && selectedPoints['x']!=undefined && selectedPoints['x'].includes(index)){
            var yIdx = 0
            selectedPoints['x'].forEach(x => {
                // console.log(x,melody_trace.y[x],selectedPoints['y'][0])
                retrainPoints.push([x,selectedPoints['y'][yIdx]])
                melody_trace.y[x] = selectedPoints["y"][yIdx];                
                yIdx++
            })
        multiple_update_melodytrace();
        }
        retrainPoints = removeDuplicateEntries(retrainPoints)
        console.log('retrain points',retrainPoints)
    }

}


function update_melodytrace(index,updatedYvalue) {
    
    melody_trace.y[index] = updatedYvalue

    // retrainPoints.push([index,updatedYvalue])

    const layout = {
        xaxis: {
                range: newRange
                } ,
        yaxis: {title: 'Frequency (Hz)',
                fixedrange: true,
                range: [0,4000],
                autorange: false,
                },
        margin: {
            b:20
        },
        
    };
    const config = {responsive: true};
    Plotly.update('js-display-spectrogram',[trace,melody_trace],layout,config);

}


function playAudio(audio) {
    // audio.currentTime = newRange[0]

    // audio.addEventListener('timeupdate', function() {
    //     if (audio.currentTime >= newRange[1]) {
    //       // Pause the audio when the end time is reached
    //       audio.pause();
    //     }
    //   });

    //   audio.play()

    if (audio.readyState >= 2) {
        audio.play()
            .then(() => {
                // Playback successful
            })
            .catch(error => {
                console.error('Error playing audio:', error);
            });
    } else {
        audio.addEventListener('loadeddata', () => {
            audio.play()
                .then(() => {
                    // Playback successful
                })
                .catch(error => {
                    console.error('Error playing audio:', error);
                });
        });
    }
}

function retrainModel() {
    // console.log('entered retrian btn')
    // console.log(retrainPoints)      
    if (retrainPoints.length!== 0) {
        let formData = new FormData()
        formData.append('file',selectedFile)
        formData.append('orig_array',JSON.stringify({data: melody_trace.y}))
        formData.append('conf_values',JSON.stringify(totalConfValues))
        formData.append('retrain_values',JSON.stringify(retrainPoints))
        // formData.append('initial_index',init_index)
        // formData.append('end_index',Math.ceil((newRange[1]).toFixed(2)*100))
        showSpinner("")           
        // Fetch to send audio data to Flask backend
        fetch("/retrain_model", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {

            retrainPoints = []
            var updated_gfv = data[0]
            // var updated_conf = data[1]
            // console.log('updated gfv',updated_gfv)


            melody_trace.y = updated_gfv
            // console.log('melody trace:',melody_trace.y)

            const layout = {
                xaxis: {range: newRange},
                yaxis: {title: 'Frequency (Hz)',
                        fixedrange: true,
                        range: [0,4000],
                        autorange: false,
                        },
                margin: {
                    b:20
                },
                
            };
            const config = {responsive: true};
            Plotly.react('js-display-spectrogram',[trace,melody_trace],layout,config);
            // Plotly.update('js-display-spectrogram',[trace,melody_trace],layout,config);


            // const filteredYValues = xValuesRange.map(x => {
            //     const index = melody_trace.x.indexOf(x);
            //     // console.log('Index:',index)
            //     return melody_trace.y[index]
            // })
            hideSpinner()
        })


    } else {
        alert('No updation!')
    }
    // console.log('Retrain completed')

}

// const marker = {
//     x: 0,
//     y: 0,
//     width:0.5,
//     height: 250,
//     speed: 0, // pixels per frame
// };

// function clearMarker() {
//     ctx2.globalCompositeOperation = 'destination-out';
//     ctx2.fillRect(marker.x, marker.y, marker.width, marker.height);
//     ctx2.globalCompositeOperation = 'source-over';
// }

// function drawMarker() {
//     ctx2.fillStyle = 'blue';
//     ctx2.fillRect(marker.x, marker.y, marker.width, marker.height);
// }

// function animateMarker() {
//     clearMarker();
//     if(stopAnimating){
//         return ;
//     }
//     // Calculate the elapsed time since animation started
//     const now = performance.now();
//     const elapsedTime = now - startTime;
//     // Update the marker's position based on time and speed
//     marker.x = (elapsedTime)/16.7 * marker.speed;
//     console.log(now,elapsedTime,marker.x,canvas2.width)
//     if (marker.x >= canvas2.width) {
//         console.log('out of canvas')
//       stopAnimation();
//       clearCanvas2();
//     } else {
//       drawMarker();
//       animationId = requestAnimationFrame(animateMarker);
//     }
// }

// function startAnimation() {
//     marker.x = 0;
//     stopAnimating = false;
//     animateMarker();
// }

// function clearCanvas2(){
//     ctx2.clearRect(0,0,canvas2.width,canvas2.height);
// }

// function stopAnimation() {
//     stopAnimating = true;
//     clearCanvas2();
//     cancelAnimationFrame(animationId);
// }




function plotSpectrogram(data) {
    trace = {
        x: data[0].x,
        y: data[0].y,
        z: data[0].z,
        showscale: true,
        colorscale: 'Viridis',
        type: 'heatmap',
        colorbar: {visible: false}
    };

    melody_trace = {
        x: data[1]['t'],
        y: data[1]['f'],
        type: 'scatter',
        mode: 'lines',
        line: {
            color: 'red' // Set the color of the lines
        },
        showlegend : false,
    }

    var overlayData = melody_trace.x
    var verticalLines = overlayData.map(function (xValue){
        return {
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: xValue,
            x1: xValue,
            y0: 0,
            y1: 1,
            line: {
                color:'gray',
                width: 0.5
            }
        }
    })

    const layout = {
        // title: 'Spectrogram',
        // xaxis: {title: 'Time(s)'},
        // shapes: verticalLines,
        yaxis: {title: 'Frequency (Hz)',
                fixedrange: true,
                range: [0,4000],
                autorange: false,
                },
        margin: {
            b:20
        },
        
    };

    const config = {responsive: true};
    Plotly.newPlot('js-display-spectrogram',[trace],layout,config);

    document.querySelector('.js-show-melody-annotate-download')
        .innerHTML = `
                <div class="show-melody-annotate">
                    <div class="show-melody"><input type="checkbox"> Show Melody</div>
                    <div class="annotate"><button class="annotate-btn">Annotate</button></div>
                    <div class="download"><button class="download-btn">Download CSV</button></div>
                </div>`

    const checkbox = document.querySelector('.js-show-melody-annotate-download input[type="checkbox"]');
    checkbox.addEventListener("change", function () {
        if (checkbox.checked) {
            // plotMelody(data[1]);
            Plotly.addTraces('js-display-spectrogram', [melody_trace]);
        } else {
            // Use Plotly.deleteTraces to remove the melody trace
            Plotly.deleteTraces('js-display-spectrogram', 1);
        }
    });

    document.querySelector('.download-btn')
        .addEventListener("click",function(){
            if (!checkbox.checked) {
                alert('Show Melody!')
            } else {
                // console.log()
                const formData = new FormData()
                formData.append('file',selectedFile)
                formData.append('freq',JSON.stringify({data: melody_trace.y}))

                fetch("/download", {
                    method: "POST",
                    body: formData
                })
                alert('Annotations Downloaded');
            }
    })

    // const combined_traces = [trace,melody_trace];

    const originalRange = document.getElementById('js-display-spectrogram')


    document.getElementById('js-display-spectrogram').on('plotly_relayout', function (eventData) {
        // Check if the x-axis range has changed
        if (eventData['xaxis.range[0]'] !== undefined && eventData['xaxis.range[1]'] !== undefined) {
            if (eventData['xaxis.range[0]'] < 0) {
                newRange = [0 , eventData['xaxis.range[1]']];

            } else {
                newRange = [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']];
                newRange = [parseFloat(eventData['xaxis.range[0]'].toFixed(2)), eventData['xaxis.range[1]']];
            }

            // console.log('new xaxis range',newRange);

            Plotly.update('js-display-spectrogram', { 'xaxis.range': newRange});            
        } 
        else {
            newRange = [Math.abs(originalRange._fullLayout.xaxis.range[0]*0),originalRange._fullLayout.xaxis.range[1]]
        }
        // console.log('newrange:',newRange)
    });

    document.querySelector('.annotate-btn')
        .addEventListener("click",function(){
            // console.log('Annotate clicked');   
            if (!checkbox.checked){
                alert('Show Melody!')
            } else {

                // console.log('Started annotation')

                showSpinner("")

                // retrainPoints = []

                if (newRange === undefined) {
                    newRange = [-1*(originalRange._fullLayout.xaxis.range[0]*0),originalRange._fullLayout.xaxis.range[1]]
                    newRange = [parseFloat(newRange[0].toFixed(2)),newRange[1]]
                }

                // console.log('New Range:',newRange);

                xValuesRange = melody_trace.x.filter(value => value >= newRange[0] && value <= newRange[1])
                xValuesRangeLength = xValuesRange.length
                // console.log('x values range:',xValuesRange)
                const filteredYValues = xValuesRange.map(x => {
                    const index = melody_trace.x.indexOf(x);
                    // console.log('Index:',index)
                    return melody_trace.y[index]
                })

                // console.log('fil y values',filteredYValues)

                const layout1 = {
                    autosize: true,
                    xaxis: {range: newRange,
                            fixedrange:true,
                            visible: false,
                            },
                    yaxis: {
                            // fixedrange:true,
                            // autosize: true,
                            autorange: true,
                            range: [0,4000],
                            visible: false,
                            },                            
                    hovermode: null,  
                    // dragmode : 'x', 
                    // shapes: verticalLines,
                    }

                
                // const zoom_trace = [trace,melody_trace];
                const zoom_trace = [trace];                    
                const config = {responsive: false};

                Plotly.newPlot('js-display-zoom-spectrogram-melody',zoom_trace,layout1,config)   
                hideSpinner("")  

                // Get confidence values
                // var confValues = randomConfValues(xValuesRange.length)                
                var confValues = getConfValues(xValuesRange.length)                
                // generateGradientPlot(confValues)

                updateChart(xValuesRange,filteredYValues)

                document.querySelector('.js-or-re-tbtn')
                    .innerHTML = `
                                    <label class="switch" id="js-or-switch"><input type="checkbox" checked> Include Original Audio <label for="volumeSliderOr"></label>
                                    <input type="range" id="volumeSliderOr" min="0" max="1" step="0.1" value="1"> 
                                    <label class="switch" id="js-re-switch"><input type="checkbox" checked> Include Resynthesize Audio <label for="volumeSliderRe" ></label>
                                    <input type="range" id="volumeSliderRe" min="0" max="1" step="0.1" value="1"> 
                                    <label class="switch" id="js-bo-switch"><input type="checkbox"> Include both serially </label>                             
                                        `

                document.querySelector('.js-play-btn')
                    .innerHTML = `<button class="play-button">Play</button>`
                document.querySelector('.js-remove-btn')
                    .innerHTML = `<button class="remove-button">Remove pitches</button>`
                document.querySelector('.js-retrain-btn')
                    .innerHTML = `<button class="retrain-button">Retrain model</button>`

                
                document.querySelector('.js-play-btn')
                    .addEventListener("click",function(){


                        // console.log('Play button clicked!!!') 
                        const volumeSliderOr = document.getElementById('volumeSliderOr');
                        const volumeSliderRe = document.getElementById('volumeSliderRe');
                        // Store last used volume
                        let lastVolumeOr = volumeSliderOr.value;
                        let lastVolumeRe = volumeSliderRe.value;

                        const formData = new FormData()
                        formData.append('file',selectedFile)
                        // formData.append('array',JSON.stringify({data: melody_trace.y}))
                        formData.append('start_time',newRange[0])
                        formData.append('end_time',newRange[1])
                        // formData.append('chunk_count',count)
                        // console.log(newRange[0],newRange[1], count)  
                        
                        const formData1 = new FormData()
                        formData1.append('file',selectedFile)
                        formData1.append('array',JSON.stringify({data: melody_trace.y}))
                        formData1.append('start_time',newRange[0])
                        formData1.append('end_time',newRange[1])

                        const fetch1 = fetch("/get_sliced_audio_original", { method: "POST", body: formData });
                        const fetch2 = fetch("/get_sliced_audio_resynth", { method: "POST", body: formData1 });

                        Promise.all([fetch1, fetch2])
                        .then(responses => Promise.all(responses.map(response => response.blob())))
                        .then(blobs => {
                            const audio1 = new Audio(URL.createObjectURL(blobs[0]));
                            const audio2 = new Audio(URL.createObjectURL(blobs[1]));

                            // Set volume to last used value
                            audio1.volume = lastVolumeOr;
                            audio2.volume = lastVolumeRe;

                            // Event listener for volume slider change for Original Audio
                            volumeSliderOr.addEventListener('input', function () {
                                audio1.volume = volumeSliderOr.value;
                            });

                            // Event listener for volume slider change for Resynthesize Audio
                            volumeSliderRe.addEventListener('input', function () {
                                audio2.volume = volumeSliderRe.value;
                            });

                            
                            var checkboxes = document.querySelectorAll('.js-or-re-tbtn input[type="checkbox"]')

                            if (checkboxes[0].checked && !checkboxes[1].checked && !checkboxes[2].checked) {
                                playAudio(audio1);
                            } else if (!checkboxes[0].checked && checkboxes[1].checked && !checkboxes[2].checked) {
                                playAudio(audio2);
                            } else if (checkboxes[0].checked && checkboxes[1].checked && !checkboxes[2].checked) {
                                playAudio(audio1);
                                playAudio(audio2);
                            } else if (checkboxes[0].checked && checkboxes[1].checked && checkboxes[2].checked) {
                                audio1.addEventListener('ended', () => {
                                    playAudio(audio2);
                                });
                                playAudio(audio1);
                            }

                        })
                        updateChart(xValuesRange,filteredYValues)
                                            
                    }) 
            
            }

            document.querySelector('.js-retrain-btn')
                .addEventListener("click",retrainModel)

        })   
    
}  //end of function plotspectrogram            


function uploadAudio() {
    const fileInput = document.getElementById('fileInput');
    fileInput.click();
    fileInput.onchange = () => {
        selectedFile = fileInput.files[0];  //const
        console.log('File:',selectedFile);

        var audioElement = new Audio();
        audioElement.src = URL.createObjectURL(selectedFile)

        audioElement.addEventListener('loadedmetadata', function () {
            // Get the duration of the audio file
            var duration = audioElement.duration;
          
            // // Check if the duration is more than 30 seconds
            // if (duration > 400) {
            //   // Show an alert
            //   alert('File duration exceeds 30 seconds. Please select a shorter file.');
            // } else {

                // Load the selected file
                wavesurfer.loadBlob(selectedFile)

                // Rewind to the beginning on finished playing (Interactive plot)
                wavesurfer.on('finish', () => {
                    wavesurfer.setTime(0) })

                wavesurfer.once('decode',()=>{
                    document.querySelector('.js-label')
                        .innerHTML = `
                        <div class="filename-label">
                            <label>Audio file: ${selectedFile['name']}</label>
                        </div>  
                        <div class="vol-zoom-playback">  
                            <div>
                                Volume: <input id="volume" type="range" min="0" max="1" step="0.1">
                            </div>  
                            <div class="zoom-label">
                                <label>Zoom: <input id="zoom" type="range" min="10" max="1000" value="100"></label>
                            </div>
                            <div>
                                <label>Playback rate: <span id="rate">1.00</span>x</label>
                                <label>0.25x <input id="playback-speed" type="range" min="0" max="4" step="1" value="2" /> 4x </label>
                                <label><input id="pitch-checkbox" type="checkbox" checked />Preserve pitch</label>
                            </div>
                        </div>`
                    document.querySelector('.js-audio-controls')
                        .innerHTML = `
                                <button class="btn-toggle-pause">
                                    <i class="fa fa-play"></i> <i class="fa fa-pause"></i>
                                </button>
                
                                <button class="btn-stop">
                                    <i class="fa fa-stop"></i>
                                </button>`

                    document.querySelector('.js-spectrogram')
                        .innerHTML = `
                                    <button class="btn-spectrogram">
                                        Show Spectrogram
                                    </button>`                           

                })

                wavesurfer.on('decode',()=>{
                    document.querySelector('.btn-toggle-pause')
                        .addEventListener("click",function(){
                            if (activeRegion){
                                // console.log('active region is present');
                                // console.log('internal',activeRegion.start,activeRegion.end)
                                if(!activeRegion.playing) {
                                    activeRegion.play();
                                } else {
                                    activeRegion.pause();
                                }
                            } else {
                                // console.log('active region not present');
                                wavesurfer.playPause();}
                            
                        })

                    document.querySelector('.btn-stop')
                        .addEventListener("click",function(){
                            wavesurfer.stop();
                        })

                    document.querySelector('#zoom')
                        .addEventListener('input', (e) => {
                            const minPxPerSec = e.target.valueAsNumber
                            wavesurfer.zoom(minPxPerSec) 
                        })

                    document.querySelector('#volume')
                        .addEventListener('input', onChangeVolume);
                    document.querySelector('#volume')
                        .addEventListener('change', onChangeVolume);

                    document.querySelector('#pitch-checkbox')
                        .addEventListener('change', (e) => {
                            preservePitch = e.target.checked
                            wavesurfer.setPlaybackRate(wavesurfer.getPlaybackRate(), preservePitch)
                        })    

                    // Set the playback rate            
                    document.querySelector('#playback-speed')
                        .addEventListener('input', (e) => {
                            const speed = speeds[e.target.valueAsNumber]
                            document.querySelector('#rate').textContent = speed.toFixed(1)
                            wavesurfer.setPlaybackRate(speed, preservePitch)
                            wavesurfer.play()
                        });  

                    document.querySelector('.js-spectrogram')
                        .addEventListener("click",function(){
                            // console.log('Spectrogram clicked')
                            showSpinner("")

                            retrainPoints = []
                            console.log('retrain points',retrainPoints)

                            let formData = new FormData();
                            formData.append("file", selectedFile);

                            
                            // Fetch to send audio data to Flask backend
                            fetch("/calculate_spectrogram", {
                                method: "POST",
                                body: formData
                            })
                            .then(response => response.json())
                            .then(data => {
                                // Use the received spectrogram data to perform any additional actions
                                // console.log('Server response',data);
                                plotSpectrogram(data)
                                hideSpinner()
                            })                   

                        });
                })

                wsRegions.enableDragSelection({
                    color: 'rgba(255,0,0,0.1)',
                })

                wsRegions.on('region-in', (region) => {
                    activeRegion = region
                    wavesurfer.play()
                    })

                wsRegions.on('region-out', (region) => {
                    wavesurfer.stop()
                    activeRegion.remove();
                    activeRegion = null
                })

                wsRegions.on('region-created', (region) => {
                    if (activeRegion){
                        activeRegion.remove();
                        activeRegion = null
                    }
                    activeRegion = region
                    // console.log('active region',activeRegion.start,activeRegion.end)     
                });

                wsRegions.on('region-clicked', (region, e) => {
                    e.stopPropagation() // prevent triggering a click on the waveform
                    activeRegion = region
                    region.play()
                    })            
                    
                // Reset the active region when the user clicks anywhere in the waveform
                wavesurfer.on('interaction', () => {
                    if (activeRegion){
                        activeRegion.remove();
                        activeRegion = null
                        wavesurfer.stop();
                        }
                    })
            }
        //   }
        );
          
          // This will trigger the 'loadedmetadata' event and initiate the process
          audioElement.load();        
        
    }
    
}



document.addEventListener('DOMContentLoaded', () => {
    const uploadButton = document.getElementById('uploadButton');

    if (uploadButton) {
        uploadButton.addEventListener('click', uploadAudio);
    }
});

