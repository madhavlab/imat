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
    // Calculate the base index once
    const init_index = parseFloat((newRange[0]).toFixed(2)) * 100;
    
    if (activePoint === null) return;
    
    // Get chart data references
    const data = activePoint._chart.data;
    const datasetIndex = activePoint._datasetIndex;
    const pointIndex = activePoint._index;
    const globalIndex = init_index + pointIndex;
    
    // Read mouse position and convert to chart value
    const position = helpers.getRelativePosition(event, myChart);
    const chartArea = myChart.chartArea;
    const yAxis = myChart.scales["y-axis-0"];
    yValue = map(position.y, chartArea.bottom, chartArea.top, yAxis.min, yAxis.max);
    
    // Ensure value is not negative
    yValue = Math.max(0, yValue);
    
    // Check if the point is part of a selection
    const isInSelection = selectedPoints && 
                         selectedPoints.x && 
                         selectedPoints.x.length > 0 && 
                         selectedPoints.x.includes(globalIndex);
    
    if (isInSelection) {
        // Calculate the change amount
        const diff = yValue - data.datasets[datasetIndex].data[pointIndex];
        
        // Update all selected points
        let yIdx = 0;
        selectedPoints.x.forEach(x => {
            const chartIndex = x - init_index;
            
            // Skip points outside the visible range
            if (chartIndex >= 0 && chartIndex < data.datasets[datasetIndex].data.length) {
                // Apply the same relative movement
                data.datasets[datasetIndex].data[chartIndex] += diff;
                data.datasets[datasetIndex].data[chartIndex] = Math.max(0, data.datasets[datasetIndex].data[chartIndex]);
                
                // Update selection cache
                selectedPoints.y[yIdx] = data.datasets[datasetIndex].data[chartIndex];
                
                // Add to retrainPoints in real-time
                retrainPoints[x] = selectedPoints.y[yIdx];
            }
            yIdx++;
        });
    } else {
        // Just update the active point
        data.datasets[datasetIndex].data[pointIndex] = yValue;
        
        // Add to retrainPoints in real-time
        retrainPoints[globalIndex] = yValue;
       
    }

    // Update canvasData to reflect new positions
    canvasData = data.labels.map((x, idx) => ({
        x: x,
        y: data.datasets[datasetIndex].data[idx]
    }));
    
    // Update the chart display
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


function handleRightClick(event) {
    event.preventDefault(); // Prevent the context menu from appearing
    
    // Clear the selection data completely
    selectedPoints = { 'x': [], 'y': [] };
    
    // Reset the box visibility flag
    boxVisible = false;
    isDragging = false;  // Also reset the dragging state
    
    // Clear any visual selection box
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    
    // Re-enable chart events
    if (myChart) {
        myChart.options.events = true;
    }
    
    // Log to confirm selection has been cleared
    console.log('Selection cleared by right-click');
    
    // Update the chart to refresh the display
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
    var temp = getPointsInsideSelectionBox();
    selectedPoints['x'] = [];
    selectedPoints['y'] = [];
    // Group the data based on the key values
    temp.forEach(item => {
      selectedPoints['x'].push(Math.round(item['x']*100));
      selectedPoints['y'].push(item['y']);
    });

    console.log('Points selected in rectangle:', 
        {
            indexes: [...selectedPoints.x], 
            values: [...selectedPoints.y],
            count: selectedPoints.x.length
        }
    );
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

function playAudio(audio) {
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
