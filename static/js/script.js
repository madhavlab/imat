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

// Add a new function to update point colors based on selection
function updatePointColors() {
    if (!myChart || !myChart.data || !myChart.data.datasets || !myChart.data.datasets[0]) return;

    const defaultColor = 'red';
    const selectedColor = 'blue';
    const pointCount = myChart.data.labels.length;

    // Default all to red
    const pointColors = new Array(pointCount).fill(defaultColor);

    // Update visible range index
    const init_index = parseInt((newRange[0] * 100).toFixed());

    // Color selected points blue
    if (selectedPoints?.x?.length > 0) {
        selectedPoints.x.forEach(globalIndex => {
            const chartIndex = globalIndex - init_index;
            if (chartIndex >= 0 && chartIndex < pointCount) {
                pointColors[chartIndex] = selectedColor;
            }
        });
    }

    // Apply new colors
    myChart.data.datasets[0].pointBackgroundColor = [...pointColors];
    myChart.data.datasets[0].pointBorderColor = [...pointColors];

    myChart.update();
}
// ####################################################################################

function map(value, start1, stop1, start2, stop2) {
    return start2 + (stop2 - start2) * ((value - start1) / (stop1 - start1))
};

function down_handler(event) {
    const points = myChart.getElementAtEvent(event,{intersect:false})
    if (points.length>0) {
        activePoint = points[0]
        canvas.onpointermove = move_handler
    } 

}

function move_handler(event) {
    if (activePoint === null) return;

    const init_index = parseInt((newRange[0] * 100).toFixed());
    const data = activePoint._chart.data;
    const datasetIndex = activePoint._datasetIndex;
    const pointIndex = activePoint._index;
    const globalIndex = init_index + pointIndex;

    const position = helpers.getRelativePosition(event, myChart);
    const chartArea = myChart.chartArea;
    const yAxis = myChart.scales["y-axis-0"];
    yValue = Math.max(0, map(position.y, chartArea.bottom, chartArea.top, yAxis.min, yAxis.max));

    const isInSelection = selectedPoints?.x?.includes(globalIndex);

    if (isInSelection) {
        updatePointColors(); // Optional: could delay this until after data update

        const diff = yValue - data.datasets[datasetIndex].data[pointIndex];
        selectedPoints.x.forEach((x, idx) => {
            const chartIndex = x - init_index;
            if (chartIndex >= 0 && chartIndex < data.datasets[datasetIndex].data.length) {
                data.datasets[datasetIndex].data[chartIndex] = Math.max(0, data.datasets[datasetIndex].data[chartIndex] + diff);
                selectedPoints.y[idx] = data.datasets[datasetIndex].data[chartIndex];
                retrainPoints[x] = selectedPoints.y[idx];
            }
        });

        updatePointColors();
    } else {
        data.datasets[datasetIndex].data[pointIndex] = yValue;
        retrainPoints[globalIndex] = yValue;
    }

    canvasData = data.labels.map((x, idx) => ({
        x: x,
        y: data.datasets[datasetIndex].data[idx]
    }));

    myChart.update();
}

function up_handler(event) {
    // First, check if activePoint is null or undefined
    if (!activePoint) return;
    
    // Then check if the expected properties exist
    if (typeof activePoint._index === 'undefined') {
        console.error('Error: activePoint lacks _index property:', activePoint);
        activePoint = null;
        canvas.onpointermove = null;
        return;
    }
    
    // Stop tracking movement
    canvas.onpointermove = null;
    
    // Calculate the base index
    const init_index = parseFloat((newRange[0]).toFixed(2)) * 100;
    const pointIndex = activePoint._index;
    const globalIndex = init_index + pointIndex;
    
    // Get the final y value (ensure it's not negative)
    const finalValue = Math.max(0, yValue || 0);
    
    // Update the retrain points and melody trace for the active point
    retrainPoints[globalIndex] = finalValue;
    update_melodytrace(globalIndex, finalValue);
    
    // Check if we're working with a selection
    const hasSelection = selectedPoints && 
                        selectedPoints.x && 
                        selectedPoints.x.length > 0 && 
                        selectedPoints.x.includes(globalIndex);
    
    // If the point is part of a selection, update ALL points in the selection
    if (hasSelection) {       
        // Update all points in the selection
        for (let i = 0; i < selectedPoints.x.length; i++) {
            const selIndex = selectedPoints.x[i];
            
            // Skip the active point, it's already been updated
            if (selIndex === globalIndex) continue;
            
            const selValue = selectedPoints.y[i];
            
            // Update both retrainPoints and melody_trace
            retrainPoints[selIndex] = selValue;
            melody_trace.y[selIndex] = selValue;
        }      
        // Call the update function once
        multiple_update_melodytrace();
    }
    
    // Update canvasData for future rectangle selections
    if (myChart && myChart.data && myChart.data.labels) {
        canvasData = myChart.data.labels.map((x, idx) => ({
            x: x,
            y: myChart.data.datasets[0].data[idx]
        }));
    }
    console.log('Total Retrain Points', retrainPoints);

    // Clear active point
    activePoint = null;
}


// HandleRightClick to reset colors when selection is cleared
function handleRightClick(event) {
    event.preventDefault();

    selectedPoints = { x: [], y: [] };
    boxVisible = false;
    isDragging = false;

    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (myChart) myChart.options.events = true;

    if (myChart?.data?.datasets?.[0]) {
        const resetColorArray = Array(myChart.data.labels.length).fill('red');
        myChart.data.datasets[0].pointBackgroundColor = [...resetColorArray];
        myChart.data.datasets[0].pointBorderColor = [...resetColorArray];
    }

    console.log('Selection cleared by right-click');
    myChart.update();
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

function adjustSelectedPoints() {
    const temp = getPointsInsideSelectionBox();
    const init_index = parseInt((newRange[0] * 100).toFixed());

    selectedPoints.x = [];
    selectedPoints.y = [];

    temp.forEach(item => {
        const globalX = Math.round(item.x * 100);  // Global X index
        selectedPoints.x.push(globalX);
        selectedPoints.y.push(item.y);
    });

    console.log('Points selected in rectangle:', {
        indexes: [...selectedPoints.x],
        values: [...selectedPoints.y],
        count: selectedPoints.x.length
    });

    updatePointColors();
}

function removeButtonClickHandler() {
    showSpinner("")
    if(selectedPoints!={} && selectedPoints['x']!=undefined){
        init_index = Math.ceil((newRange[0]).toFixed(2)*100)   
        var yIdx = 0
        selectedPoints['x'].forEach(x => {
            myChart.data.datasets[0].data[x-init_index] = 0
            selectedPoints['y'][yIdx] = 0 
            retrainPoints[x] = 0
            yIdx++
        })            
    }
    console.log('Total Retrain Points',retrainPoints)

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
        if (boxVisible == false) {
            isDragging = true;
            boxVisible = true;
            myChart.options.events = false;
            selectionStart = helpers.getRelativePosition(event, myChart)
            selectionEnd = {...selectionStart}
        }
    })

    canvas.addEventListener('contextmenu',handleRightClick);

    canvas.addEventListener('mousemove',(event)=>{
        if (isDragging) {
            selectionEnd = helpers.getRelativePosition(event, myChart)
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


// ####################################################################################

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


function multiple_update_melodytrace() {
    const layout = {
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


// ####################################################################################

function updateChart(xValuesRange,filteredYValues) {
    chartData = {
        labels: xValuesRange,
        datasets: [{
            data: filteredYValues,
            label: ' ',
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
                maintainAspectRatio: false,
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

// ####################################################################################

function createTimeMarker() {
    // Check if marker already exists and remove it
    const existingMarker = document.getElementById('time-marker');
    if (existingMarker) {
        existingMarker.remove();
    }
    
    // Create the marker element
    const marker = document.createElement('div');
    marker.id = 'time-marker';
    marker.style.position = 'absolute';
    marker.style.top = '0';
    marker.style.width = '2px';
    marker.style.height = '100%';
    marker.style.backgroundColor = 'red';
    marker.style.zIndex = '1000';
    marker.style.pointerEvents = 'none';
    marker.style.transition = 'left 0.1s linear';
    
    // Add it to the chart container
    const container = document.querySelector('.chart-container');
    if (container) {
        container.appendChild(marker);
        return marker;
    }
    return null;
}

function updateTimeMarkerPosition(marker, startTime, currentTime, endTime, container) {
    if (!marker || !container || !myChart) return;
    
    // Calculate the current time value
    const timeValue = currentTime;
    
    // Convert the time value to x-coordinate using the chart's scales
    // This ensures proper alignment with data points
    const xScale = myChart.scales['x-axis-0'];
    
    // Find the closest x value in our data points
    const closestXValue = findClosestXValue(timeValue);
    
    // Use the chart's scale to get the pixel position
    const pixelPosition = xScale.getPixelForValue(closestXValue);
    
    // Set the marker position directly
    marker.style.left = pixelPosition + 'px';
}

// Helper function to find the closest x value in our data
function findClosestXValue(timeValue) {
    // If we don't have xValuesRange, return the time value
    if (!xValuesRange || xValuesRange.length === 0) return timeValue;
    
    // Find the closest match
    return xValuesRange.reduce((prev, curr) => 
        Math.abs(curr - timeValue) < Math.abs(prev - timeValue) ? curr : prev
    );
}

function playAudio(audio, startTime, endTime) {
    const marker = createTimeMarker();
    const container = document.querySelector('.chart-container');
    
    // Position marker initially at the first data point, not at the beginning of the chart
    if (marker && myChart && xValuesRange && xValuesRange.length > 0) {
        const firstDataPoint = xValuesRange[0]; // The first x value in our visible range
        const xScale = myChart.scales['x-axis-0'];
        const initialPixelPosition = xScale.getPixelForValue(firstDataPoint);
        marker.style.left = initialPixelPosition + 'px';
    }
    
    // Animation variables
    let animationFrameId = null;
    let lastTime = 0;
    
    // Animation function using requestAnimationFrame for smoother updates
    const animateMarker = (timestamp) => {
        if (!lastTime) lastTime = timestamp;
        
        // Only update every 30ms for efficiency
        if (timestamp - lastTime >= 30) {
            updateTimeMarkerPosition(marker, startTime, audio.currentTime + startTime, endTime, container);
            lastTime = timestamp;
        }
        
        if (!audio.paused && !audio.ended) {
            animationFrameId = requestAnimationFrame(animateMarker);
        } else if (audio.ended) {
            if (marker) marker.style.display = 'none';
        }
    };
    
    // Add event listeners
    audio.addEventListener('play', () => {
        if (marker) marker.style.display = 'block';
        animationFrameId = requestAnimationFrame(animateMarker);
    });
    
    audio.addEventListener('pause', () => {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
    });
    
    audio.addEventListener('ended', () => {
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
            animationFrameId = null;
        }
        if (marker) marker.style.display = 'none';
    });
    
    if (audio.readyState >= 2) {
        audio.play()
            .then(() => {
                // We've already positioned the marker at the first data point,
                // no need to update the initial position here
            })
            .catch(error => {
                console.error('Error playing audio:', error);
                if (marker) marker.style.display = 'none';
            });
    } else {
        audio.addEventListener('loadeddata', () => {
            audio.play()
                .then(() => {
                    // We've already positioned the marker at the first data point
                })
                .catch(error => {
                    console.error('Error playing audio:', error);
                    if (marker) marker.style.display = 'none';
                });
        });
    }
}

// ####################################################################################

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

function generateGradientPlot(conf_frame_val){
    var gradientColors = generateGradientColors(conf_frame_val);
    var gradientBar = document.getElementById('js-confidence-bar');
    // gradientBar.style.background = 'none';
    gradientBar.style.background = 'linear-gradient(to right, ' + gradientColors.join(', ') + ')';
}

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
        generateGradientPlot(cValues)
        hideSpinner()
    })
}


function retrainModel() {
    if (Object.keys(retrainPoints).length !== 0) {
        // Convert retrainPoints from object format to array format
        const retrainPointsArray = Object.entries(retrainPoints).map(([index, value]) => {
            return [parseInt(index), value];
        });
        console.log('Retrain Array points',retrainPointsArray)       
        let formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('orig_array', JSON.stringify({data: melody_trace.y}))
        formData.append('conf_values', JSON.stringify(totalConfValues))
        formData.append('retrain_values', JSON.stringify(retrainPointsArray))
        
        showSpinner("")           
        
        fetch("/retrain_model", {
            method: "POST",
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            retrainPoints = {}; // Reset points after success
            var updated_gfv = data[0];
            melody_trace.y = updated_gfv;

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
            hideSpinner();
        })
        .catch(error => {
            console.error('Error retraining model:', error);
            alert(`Error retraining model: ${error.message}`);
            hideSpinner();
        });
    } else {
        alert('No points to update!');
    }
}

// ####################################################################################
// Plot Spectogram and make it interactive

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
            Plotly.addTraces('js-display-spectrogram', [melody_trace]);
        } else {
            Plotly.deleteTraces('js-display-spectrogram', 1);
        }
    });

    // Download the pitch by clicking on the Download CSV button
    document.querySelector('.download-btn')
        .addEventListener("click",function(){
            if (!checkbox.checked) {
                alert('Show Melody!')
            } else {
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

    const originalRange = document.getElementById('js-display-spectrogram')

    // Zoom into the spectrogram and update the Plotly plot
    document.getElementById('js-display-spectrogram').on('plotly_relayout', function (eventData) {
        if (eventData['xaxis.range[0]'] !== undefined && eventData['xaxis.range[1]'] !== undefined) {
            if (eventData['xaxis.range[0]'] < 0) {
                newRange = [0 , eventData['xaxis.range[1]']];

            } else {
                newRange = [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']];
                newRange = [parseFloat(eventData['xaxis.range[0]'].toFixed(2)), eventData['xaxis.range[1]']];
            }
            Plotly.update('js-display-spectrogram', { 'xaxis.range': newRange});            
        } 
        else {
            newRange = [Math.abs(originalRange._fullLayout.xaxis.range[0]*0),originalRange._fullLayout.xaxis.range[1]]
        }
    });

    document.querySelector('.annotate-btn')
    .addEventListener("click", function() {
        if (!checkbox.checked) {
            alert('Show Melody!')
        } else {
            showSpinner("")

            // Fix for undefined newRange
            if (newRange === undefined || newRange === null) {
                // Get the default range from the spectrogram
                if (originalRange && originalRange._fullLayout && originalRange._fullLayout.xaxis) {
                    newRange = [0, originalRange._fullLayout.xaxis.range[1]];
                } else {
                    // Fallback if nothing else is available
                    newRange = [0, Math.max(...melody_trace.x)];
                }
                newRange = [parseFloat(newRange[0].toFixed(2)), newRange[1]];
                console.log("Using default range:", newRange);
            }
           
            xValuesRange = melody_trace.x.filter(value => value >= newRange[0] && value <= newRange[1])
            xValuesRangeLength = xValuesRange.length
            const filteredYValues = xValuesRange.map(x => {
                const index = melody_trace.x.indexOf(x);
                return melody_trace.y[index]
            })

            const layout1 = {
                autosize: true,
                xaxis: {range: newRange,
                        fixedrange: true,
                        visible: false,
                        },
                yaxis: {
                        autorange: true,
                        range: [0, 4000],
                        visible: false,
                        },                            
                hovermode: null,  
                }
            const zoom_trace = [trace];                    
            const config = {responsive: true};

            Plotly.newPlot('js-display-zoom-spectrogram-melody', zoom_trace, layout1, config)   
            hideSpinner("") 

            var confValues = getConfValues(xValuesRange.length)                

            updateChart(xValuesRange, filteredYValues)

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
                .addEventListener("click", function() {
                    const volumeSliderOr = document.getElementById('volumeSliderOr');
                    const volumeSliderRe = document.getElementById('volumeSliderRe');
                    // Store last used volume
                    let lastVolumeOr = volumeSliderOr.value;
                    let lastVolumeRe = volumeSliderRe.value;

                    const formData = new FormData()
                    formData.append('file', selectedFile)
                    formData.append('start_time', newRange[0])
                    formData.append('end_time', newRange[1])
                    
                    const formData1 = new FormData()
                    formData1.append('file', selectedFile)
                    formData1.append('array', JSON.stringify({data: melody_trace.y}))
                    formData1.append('start_time', newRange[0])
                    formData1.append('end_time', newRange[1])

                    const fetch1 = fetch("/get_sliced_audio_original", { method: "POST", body: formData });
                    const fetch2 = fetch("/get_sliced_audio_resynth", { method: "POST", body: formData1 });

                    const startTime = newRange[0];
                    const endTime = newRange[1];
                    
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
                            playAudio(audio1, startTime, endTime);
                        } else if (!checkboxes[0].checked && checkboxes[1].checked && !checkboxes[2].checked) {
                            playAudio(audio2, startTime, endTime);
                        } else if (checkboxes[0].checked && checkboxes[1].checked && !checkboxes[2].checked) {
                            playAudio(audio1, startTime, endTime);
                            playAudio(audio2, startTime, endTime);
                        } else if (checkboxes[0].checked && checkboxes[1].checked && checkboxes[2].checked) {
                            audio1.addEventListener('ended', () => {
                                playAudio(audio2, startTime, endTime);
                            });
                            playAudio(audio1, startTime, endTime);
                        }

                    })
                    updateChart(xValuesRange, filteredYValues)
                                        
                }) 
        }
            document.querySelector('.js-retrain-btn')
                .addEventListener("click",retrainModel)

        })   

}


// ####################################################################################
// Upload an audio and display the waveform

// Store references to event listeners for cleanup
const eventListenerReferences = {
    wavesurfer: [],
    regions: [],
    ui: []
};

// Initiates the audio file upload process

function uploadAudio() {
    const fileInput = document.getElementById('fileInput');
    fileInput.click();
    fileInput.onchange = handleFileSelection;
}


// Handles the file selection from the input

function handleFileSelection() {
    const fileInput = document.getElementById('fileInput');
    selectedFile = fileInput.files[0];
    console.log('File:', selectedFile);
    
    // Validate the file before processing
    validateAudioFile(selectedFile);
}


// Validates the audio file before processing

function validateAudioFile(file) {
    const audioElement = new Audio();
    audioElement.src = URL.createObjectURL(file);
    
    audioElement.addEventListener('loadedmetadata', function() {
        const duration = audioElement.duration;
        const MAX_DURATION = 30; // seconds
        
        if (duration > MAX_DURATION) {
            alert(`File duration exceeds ${MAX_DURATION} seconds. Please select a shorter file.`);
        } else {
            // Clean up existing resources before processing new file
            cleanupExistingResources();
            
            // Process the audio file
            processAudioFile(file);
        }
    });
    
    // Error handling for audio loading
    audioElement.addEventListener('error', function() {
        alert('Error loading audio file. Please try another file.');
    });
    
    audioElement.load();
}

// Processes a valid audio file
function processAudioFile(file) {
    // Load the selected file into wavesurfer
    wavesurfer.loadBlob(file);
    
    // Set up event handlers
    setupWavesurferEvents();
    setupRegionsEvents();
}


// Sets up WaveSurfer-related event handlers
function setupWavesurferEvents() {
    // Rewind to beginning when finished playing
    const finishHandler = () => {
        wavesurfer.setTime(0);
    };
    wavesurfer.on('finish', finishHandler);
    eventListenerReferences.wavesurfer.push({ event: 'finish', handler: finishHandler });
    
    // Handle decoding completion
    wavesurfer.once('decode', setupUserInterface);
    
    // Reset active region on waveform interaction
    const interactionHandler = () => {
        if (activeRegion) {
            activeRegion.remove();
            activeRegion = null;
            wavesurfer.stop();
        }
    };
    wavesurfer.on('interaction', interactionHandler);
    eventListenerReferences.wavesurfer.push({ event: 'interaction', handler: interactionHandler });
}

// Sets up region-related event handlers
function setupRegionsEvents() {
    // Enable region selection
    wsRegions.enableDragSelection({
        color: 'rgba(255,0,0,0.1)',
    });
    
    // Region event handlers
    const regionInHandler = (region) => {
        activeRegion = region;
        wavesurfer.play();
    };
    wsRegions.on('region-in', regionInHandler);
    eventListenerReferences.regions.push({ event: 'region-in', handler: regionInHandler });
    
    const regionOutHandler = (region) => {
        wavesurfer.stop();
        activeRegion.remove();
        activeRegion = null;
    };
    wsRegions.on('region-out', regionOutHandler);
    eventListenerReferences.regions.push({ event: 'region-out', handler: regionOutHandler });
    
    const regionCreatedHandler = (region) => {
        if (activeRegion) {
            activeRegion.remove();
            activeRegion = null;
        }
        activeRegion = region;
    };
    wsRegions.on('region-created', regionCreatedHandler);
    eventListenerReferences.regions.push({ event: 'region-created', handler: regionCreatedHandler });
    
    const regionClickedHandler = (region, e) => {
        e.stopPropagation(); // prevent triggering a click on the waveform
        activeRegion = region;
        region.play();
    };
    wsRegions.on('region-clicked', regionClickedHandler);
    eventListenerReferences.regions.push({ event: 'region-clicked', handler: regionClickedHandler });
}


// Sets up the user interface elements
function setupUserInterface() {
    setupFileInfoUI();
    setupAudioControlsUI();
    setupSpectrogramUI();
    attachUIEventListeners();
}


// Sets up the file information UI elements

function setupFileInfoUI() {
    document.querySelector('.js-label').innerHTML = `
        <div class="filename-label">
            <label>Audio file: ${selectedFile.name}</label>
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
        </div>`;
}

//  Sets up the audio controls UI elements

function setupAudioControlsUI() {
    document.querySelector('.js-audio-controls').innerHTML = `
        <button class="btn-toggle-pause">
            <i class="fa fa-play"></i> <i class="fa fa-pause"></i>
        </button>
        <button class="btn-stop">
            <i class="fa fa-stop"></i>
        </button>`;
}

// Sets up the spectrogram UI elements

function setupSpectrogramUI() {
    document.querySelector('.js-spectrogram').innerHTML = `
        <button class="btn-spectrogram">
            Show Spectrogram
        </button>`;
}

function onChangeVolume(e) {
    const volume = e.target.valueAsNumber;
    wavesurfer.setVolume(volume);
}

// Attaches event listeners to UI elements

function attachUIEventListeners() {
    // Play/pause button
    const playPauseBtn = document.querySelector('.btn-toggle-pause');
    if (playPauseBtn) {
        const playPauseHandler = function() {
            if (activeRegion) {
                if (!activeRegion.playing) {
                    activeRegion.play();
                } else {
                    activeRegion.pause();
                }
            } else {
                wavesurfer.playPause();
            }
        };
        playPauseBtn.addEventListener("click", playPauseHandler);
        eventListenerReferences.ui.push({ element: playPauseBtn, event: 'click', handler: playPauseHandler });
    }
    
    // Stop button
    const stopBtn = document.querySelector('.btn-stop');
    if (stopBtn) {
        const stopHandler = () => wavesurfer.stop();
        stopBtn.addEventListener("click", stopHandler);
        eventListenerReferences.ui.push({ element: stopBtn, event: 'click', handler: stopHandler });
    }
    
    // Zoom control
    const zoomControl = document.querySelector('#zoom');
    if (zoomControl) {
        const zoomHandler = (e) => {
            const minPxPerSec = e.target.valueAsNumber;
            wavesurfer.zoom(minPxPerSec);
        };
        zoomControl.addEventListener('input', zoomHandler);
        eventListenerReferences.ui.push({ element: zoomControl, event: 'input', handler: zoomHandler });
    }
    
    // Volume controls
    const volumeControl = document.querySelector('#volume');
    if (volumeControl) {
        volumeControl.addEventListener('input', onChangeVolume);
        volumeControl.addEventListener('change', onChangeVolume);
        eventListenerReferences.ui.push({ element: volumeControl, event: 'input', handler: onChangeVolume });
        eventListenerReferences.ui.push({ element: volumeControl, event: 'change', handler: onChangeVolume });
    }
    
    // Pitch preservation checkbox
    const pitchCheckbox = document.querySelector('#pitch-checkbox');
    if (pitchCheckbox) {
        const pitchHandler = (e) => {
            preservePitch = e.target.checked;
            wavesurfer.setPlaybackRate(wavesurfer.getPlaybackRate(), preservePitch);
        };
        pitchCheckbox.addEventListener('change', pitchHandler);
        eventListenerReferences.ui.push({ element: pitchCheckbox, event: 'change', handler: pitchHandler });
    }
    
    // Playback speed control
    const playbackSpeedControl = document.querySelector('#playback-speed');
    if (playbackSpeedControl) {
        const playbackSpeedHandler = (e) => {
            const speed = speeds[e.target.valueAsNumber];
            document.querySelector('#rate').textContent = speed.toFixed(1);
            wavesurfer.setPlaybackRate(speed, preservePitch);
            wavesurfer.play();
        };
        playbackSpeedControl.addEventListener('input', playbackSpeedHandler);
        eventListenerReferences.ui.push({ element: playbackSpeedControl, event: 'input', handler: playbackSpeedHandler });
    }
    
    // Spectrogram button
    const spectrogramBtn = document.querySelector('.btn-spectrogram');
    if (spectrogramBtn) {
        const spectrogramHandler = function() {
            showSpinner("");
                       
            let formData = new FormData();
            formData.append("file", selectedFile);
            
            // Fetch to send audio data to Flask backend
            fetch("/calculate_spectrogram", {
                method: "POST",
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                plotSpectrogram(data);
                hideSpinner();
            })
            .catch(error => {
                console.error('Error generating spectrogram:', error);
                alert('Error generating spectrogram. Please try again.');
                hideSpinner();
            });
        };
        spectrogramBtn.addEventListener("click", spectrogramHandler);
        eventListenerReferences.ui.push({ element: spectrogramBtn, event: 'click', handler: spectrogramHandler });
    }
}

// Cleans up existing resources before loading a new file

function cleanupExistingResources() {
   // Clean up wavesurfer event listeners
   eventListenerReferences.wavesurfer.forEach(item => {
       wavesurfer.un(item.event, item.handler);
   });
   eventListenerReferences.wavesurfer = [];
   
   // Clean up region event listeners
   eventListenerReferences.regions.forEach(item => {
       wsRegions.un(item.event, item.handler);
   });
   eventListenerReferences.regions = [];
   
   // Clean up UI event listeners
   eventListenerReferences.ui.forEach(item => {
       if (item.element) {
           item.element.removeEventListener(item.event, item.handler);
       }
   });
   eventListenerReferences.ui = [];
   
   // Remove all existing regions
   if (wsRegions) {
       wsRegions.clearRegions();
   }
   
   // Reset active region
   activeRegion = null;
   
   // Reset cursor position to beginning
   if (wavesurfer) {
       wavesurfer.stop(); // This stops playback
       wavesurfer.setTime(0); // This explicitly sets the cursor to the beginning
   }
}


// ####################################################################################
document.addEventListener('DOMContentLoaded', () => {
    const uploadButton = document.getElementById('uploadButton');

    if (uploadButton) {
        uploadButton.addEventListener('click', uploadAudio);
    }
});



