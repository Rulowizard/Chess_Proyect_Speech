////////////////////////////////////////////////////////////////////////////////////////
//Sección para variables globales
min_val=1
max_val=833
var bar_datos;
var hist_datos;
var heat_data;
var box_data_time;
var box_data_len;

var global_width = 700;
var global_height = 700;

////////////////////////////////////////////////////////////////////////////////////////
//Sección para funciones filtro
function selectDepth0(data){
    return data.depth == 0
}

function selectDepth1(data){
    return data.depth == 1
}

function selectDepth2(data){
    return data.depth == 2
}

function selectEndGame(data){
    return data.winner != ""
}

function selectM1(data){
    return data.playerplays_type == "M1"
}

function selectM2(data){
    return data.playerplays_type == "M2"
}

function selectM3(data){
    return data.playerplays_type == "M3"
}

function selectM4(data){
    return data.playerplays_type == "M4"
}

function selectM5(data){
    return data.playerplays_type == "M5"
}

function selectM6(data){
    return data.playerplays_type == "M6"
}

function selectH(data){
    return data.playerplays_type == "H"
}

function selectMinTurn(data){
    return data.game_len >= min_val
}

function selectMaxTurn(data){
    return data.game_len<= max_val
}

////////////////////////////////////////////////////////////////////////////////////////
//Sección para funciones filtro de datos

function filterDataByDepth(data){
    var valor = getSelDepth()

    //Filtrar por profundidad
    switch(valor){
        case "Depth 1":
            filtered_data = data.filter(selectDepth1);
            break;
        case "Depth 2":
            //obtener info filtrada
            filtered_data = data.filter(selectDepth2);
            break;
        case "Depth 0":
            //obtener info filtrada
            filtered_data = data.filter(selectDepth0);
            break;
        default:
            //obtener info filtrada
            filtered_data = data
            break;
    }

    return filtered_data;
}

function filterDataByPlayer(filtered_data){
    var player = getSelPlayer()

    //Filtrar por jugador
    switch(player){
        case "Machine 1":
            filtered_data = filtered_data.filter(selectM1);
            break;
        case "Machine 2":
            filtered_data = filtered_data.filter(selectM2);
            break;
        case "Machine 3":
            filtered_data = filtered_data.filter(selectM3);
            break;
        case "Machine 4":
            filtered_data = filtered_data.filter(selectM4);
            break;
        case "Machine 5":
            filtered_data = filtered_data.filter(selectM5);
            break;
        case "Machine 6":
            filtered_data = filtered_data.filter(selectM6);
            break;
        case "Humano":
            filtered_data = filtered_data.filter(selectH);
            break;
    }
    return filtered_data;
}


////////////////////////////////////////////////////////////////////////////////////////
//Seccion para funciones que extraen valores de las listas de selección
function getSelDepth(){
    return d3.select("#selDepth").property("value")
}

function getSelPlayer(){
    return d3.select("#selPlayer").property("value")
}

////////////////////////////////////////////////////////////////////////////////////////
//Sección para las funciones que dibujan gráficas
function drawBarWinner(data=bar_datos){
    var player = getSelPlayer()
    var info_arr=[]

    var layout={
        width:global_width,
        height:global_height,
        title:"Bar Chart - Type of End Game Condition",
        barmode:"stack"
    }

    switch(player){
        case "All":
            var t1 ={
                y: data[0][1],
                x: data[0][0],
                name:"M1",
                type:"bar"
            };
            var t2 ={
                y: data[1][1],
                x: data[1][0],
                name:"M2",
                type:"bar"
            };
            var t3 ={
                y: data[2][1],
                x: data[2][0],
                name:"M3",
                type:"bar"
            };
            var t4 ={
                y: data[3][1],
                x: data[3][0],
                name:"M4",
                type:"bar"
            };
            var t5 ={
                y: data[4][1],
                x: data[4][0],
                name:"M5",
                type:"bar"
            };
            var t6 ={
                y: data[5][1],
                x: data[5][0],
                name:"M6",
                type:"bar"
            };
            var t7 ={
                y: data[6][1],
                x: data[6][0],
                name:"H",
                type:"bar"
            };
            info_arr=[t1,t2,t3,t4,t5,t6,t7];
            //var layout = {barmode:"stack"};
            Plotly.newPlot("img-2",info_arr,layout)
        break;

        case "Machine 1":
            var t ={
                y: data[0][1],
                x: data[0][0],
                type:"bar"
            };
            info_arr=[t];
            Plotly.newPlot("img-2",info_arr,layout)
        break;

        case "Machine 2":
            var t ={
                y: data[1][1],
                x: data[1][0],
                type:"bar"
            };
            info_arr=[t];
            Plotly.newPlot("img-2",info_arr,layout)
        break;

        case "Machine 3":
            var t ={
                y: data[2][1],
                x: data[2][0],
                type:"bar"
            };
            info_arr=[t];
            Plotly.newPlot("img-2",info_arr,layout)
        break;

        case "Machine 4":
            var t ={
                y: data[3][1],
                x: data[3][0],
                type:"bar"
            };
            info_arr=[t];
            Plotly.newPlot("img-2",info_arr,layout)
        break;

        case "Machine 5":
            var t ={
                y: data[4][1],
                x: data[4][0],
                type:"bar"
            };
            info_arr=[t];
            Plotly.newPlot("img-2",info_arr,layout)
        break;

        case "Machine 6":
            var t ={
                y: data[5][1],
                x: data[5][0],
                type:"bar"
            };
            info_arr=[t];
            Plotly.newPlot("img-2",info_arr,layout)
        break;

        case "Humano":
            var t ={
                y: data[6][1],
                x: data[6][0],
                type:"bar"
            };
            info_arr=[t];
            Plotly.newPlot("img-2",info_arr,layout)
        break;
    }
}

function drawHistogramPosEvaluated(data=hist_datos){
    //Filtrar info usando depth y jugador
    filtered_data = filterDataByDepth(data)
    filtered_data = filterDataByPlayer(filtered_data)

    //Filtrar por turno
    filtered_data = filtered_data.filter(selectMaxTurn)
    filtered_data = filtered_data.filter(selectMinTurn)

    //Preparar info
    var hist_mov_len = {
        x: filtered_data.map( row => row.mov_len ),
        type: "histogram",
    };
    //Dibujar histograma
    var hist_info = [hist_mov_len];

    var layout={
        width:global_width,
        height:global_height,
        title:"Histogram - Positions Analyzed",
        barmode:"stack"
    }

    Plotly.newPlot("img", hist_info, layout)
}

function drawHeatMap(data=heat_data){
    //Filtrar info usando depth y jugador
    filtered_data = filterDataByDepth(data)
    filtered_data = filterDataByPlayer(filtered_data)
    //Filtrar por turno
    filtered_data = filtered_data.filter(selectMaxTurn)
    filtered_data = filtered_data.filter(selectMinTurn)

    var heat_info = {
        x: filtered_data.map( row => row.x_dest ),
        y: filtered_data.map(row=> row.y_dest ),
        type: "histogram2d",
        xbins:{
            start:1,
            end:9,
            size:1
        },
        ybins:{
            start:1,
            end:9,
            size:1
        }
    }

    var layout ={
        width:global_width,
        height:global_height,
        title:"Heatmap - Played Positions",
        xaxis:{
            tickvals:[1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5],
            ticktext:["a","b","c","d","e","f","g","h"]
        },
        yaxis:{
            tickvals:[1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5],
            ticktext:[1,2,3,4,5,6,7,8]
        }
    }
    Plotly.newPlot("img-3",[heat_info] , layout )
}


function drawBoxTime(data = box_data_time){
    //Filtrar info usando jugador
    filtered_data = filterDataByPlayer(data)
    filtered_data = filterDataByDepth(filtered_data)
    //Filtrar por turno
    filtered_data = filtered_data.filter(selectMaxTurn)
    filtered_data = filtered_data.filter(selectMinTurn)

    //data.filter(selectDepth1);
    var box_info_0={
        y: filtered_data.filter(selectDepth0).map(row=> row.exec_time),
        type:"box",
        boxpoints: false,
        name:"Depth 0" 
    };

    var box_info_1={
        y: filtered_data.filter(selectDepth1).map(row=> row.exec_time),
        type:"box",
        boxpoints: false,
        name:"Depth 1" 
    };

    var box_info_2={
        y: filtered_data.filter(selectDepth2).map(row=> row.exec_time),
        type:"box",
        boxpoints: false,
        name:"Depth 2" 
    };

    var layout={
        width:global_width,
        height:global_height,
        title:"Box Plot - Execution Time per Move"
    }

    box_info=[box_info_0,box_info_1,box_info_2]
    Plotly.newPlot("img-4",box_info, layout)
}

function drawBoxLen(data = box_data_len){
    //Filtrar info usando jugador
    filtered_data = filterDataByPlayer(data)
    filtered_data = filterDataByDepth(filtered_data)
    //Filtrar por turno
    filtered_data = filtered_data.filter(selectMaxTurn)
    filtered_data = filtered_data.filter(selectMinTurn)

    //data.filter(selectDepth1);
    var box_info_0={
        y: filtered_data.filter(selectDepth0).map(row=> row.mov_len),
        type:"box",
        boxpoints: false,
        name: "Depth 0" 
    };

    var box_info_1={
        y: filtered_data.filter(selectDepth1).map(row=> row.mov_len),
        type:"box",
        boxpoints: false,
        name:"Depth 1" 
    };

    var box_info_2={
        y: filtered_data.filter(selectDepth2).map(row=> row.mov_len),
        type:"box",
        boxpoints: false,
        name:"Depth 2" 
    };

    var layout={
        width:global_width,
        height:global_height,
        title:"Box Plot - Number of Positions Analyzed"
    }

    box_info=[box_info_0,box_info_1,box_info_2]
    Plotly.newPlot("img-5",box_info, layout)
}

function init(){
    $.get("box_time", function(data){
        box_data_time = data;
        drawBoxTime(box_data_time);
    });

    $.get("box_len", function(data){
        box_data_len = data;
        drawBoxLen(box_data_len);
    });

    $.get("bars", {min:1,max:633,depth:getSelDepth(),player:getSelPlayer()} , function(data){
        bar_datos= data
        drawBarWinner(bar_datos)
    });

    $.get("mysql", function(data){
        hist_datos = data
        drawHistogramPosEvaluated(hist_datos)
    });

    $.get("heat_map", function(data){
        heat_data = data
        drawHeatMap(heat_data)
    })
}

function slider(data){
    min_val = data[0]
    max_val = data[1]
    drawHistogramPosEvaluated(hist_datos)
    drawHeatMap();
    drawBarWinner();
    drawBoxTime();
    drawBoxLen();
}

d3.select("#selDepth").on("change", function(){
    drawHistogramPosEvaluated();
    drawBarWinner();
    drawHeatMap();
    drawBoxTime();
    drawBoxLen();
} );

d3.select("#selPlayer").on("change", function(){
    drawHistogramPosEvaluated();
    drawBarWinner();
    drawHeatMap();
    drawBoxTime();
    drawBoxLen();
} );


init();

