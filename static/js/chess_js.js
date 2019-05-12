///////////////////////////////////////
//Javascript para pagina web test_svg//
///////////////////////////////////////

//Variables globales
//Color define a quien le toca mover. True es blanco
var color=true;
//Verdadero si no hay jaque mate o alguna condicion de mepate
var continuar = true;
//Arreglo que guardara los clicks del jugador humano
var clicks =[];
//Variable que guarda la longitud del juego
var game_len=0;
//Variabele que guarda forma en que acabo el juego
var msg="";
//Variable que guarda el ganador de la partida
var result="";
//Variable que guarda el id del juego
var game_id = "";


//Inicializa en el lado del servidor un tablero nuevo
//No regresa el tablero pero si manda mensaje
function initialize(){
  color=true;
  continuar=true;
  game_len=0;
  msg="";
  result="";
  var contenedor = d3.select("#contenedor-text");
  //Quitar todos los elementos del contenedor
  contenedor.selectAll("*").remove();
  var row = document.getElementById("row-render")
  if (row != null){
    console.log("Not Null")
    d3.selectAll("#row-render").remove()
  }
  

  $.get("initialize",function(data){
    game_id = parseInt(data)
    handleSubmit()
  });

}

//Modificar HTML para mostrar estadisticas basicas del juego
function game_stats(data){
  //Seleccionar el contenedor
  var contenedor = d3.select("#contenedor-text");
  //Quitar todos los elementos del contenedor
  contenedor.selectAll("*").remove();
  //Agregar titulo
  var h2s = contenedor.append("h1").attr("class","titulo")
  h2s.text("Game Information")
  //Agregar espacio
  var br1 =contenedor.append("br")
  //Agregar texto
  var p1 = contenedor.append("p").attr("class","stat-text").text("Turno: "+data[2] )
  var p2 = contenedor.append("p").attr("class","stat-text").text("Mueve: " + String( (color==true) ? "Blanco" :"Negro") )
  var p5 = contenedor.append("p").attr("class","stat-text").text("Tiempo de Procesamiento: " + String(data[4]).substring(0,10) )
  var p6 = contenedor.append("p").attr("class","stat-text").text("Score: " + String(data[3]).substring(0,9) )
  var p7 = contenedor.append("p").attr("class","stat-text").text("Posiciones Analizadas: " + String(data[5]) )

  if(data[6]!=""){
    var p3= contenedor.append("p").attr("class","stat-text").text("Resultado: " + data[6] )
    var p4= contenedor.append("p").attr("class","stat-text").text("Ganador: " +data[7])
  }
}


//Regresa imagen SVG y valor si continua el juego
//Sólo para máquinas
//Recibe valor de la lista de seleccion del jugador actual y profundidad
async function getInfo(player,depth){
  var info="";

  info= await $.get("player",{player:player,depth:depth,game_id:game_id},function(data){
    //Data[0] viene info sobre la imagen SVG
    //Data[1] es la variable bool que indica si continua el juego
    
    //Convierte el string en objeto
    var obj_svg=$(data[0])
    //Acceder a la info de la imagen
    var img_svg=obj_svg[0]

    game_stats(data);

    return {cont:data[1],img:data[0]};
  }).then(d => d);

  return info;

}

//Regresa el valor de profundidad de la lista de seleccion para el jugador actual
//usa la variable global de color
function getSelDepth(){
  if (color==true){
    var sj = d3.select("#depth1")
  }else{
    var sj = d3.select("#depth2")
  }
  return sj.property("value");
}


//Regresa el valor de la lista de seleccion para el jugador actual
//usa la variable global de color
function getSelPlayer(){
  if (color==true){
    var sj = d3.select("#selPlayer1");
  } else{
    var sj = d3.select("#selPlayer2");
  }
  return sj.property("value");
}

//Regresa la equivalencia del lado del servidor del valor de la lista de seleccion
function serverSelPlayer(selValue){

  switch(selValue){
    case "Maquina 1":
      player="M1"
      break;
    case "Maquina 2":
      player="M2"
      break;
    case "Maquina 3":
      player="M3"
      break;
    case "Maquina 4":
      player="M4"
      break;
    case "Humano":
      player="Humano"
      break;
    case "Maquina 5":
      player="M5"
      break;
    default:
      player="M6"
  }
  return player;

}

//Dibujar una imagen SVG en la página web
function dibujarSVG(img){
  new_svg = $(img)
  var svg = document.getElementById("svg");
  svg.innerHTML="";
  svg.appendChild(new_svg[0]);
  var img = d3.select("svg")
  img.attr("height",500)
  img.attr("width",500)
}

//Proceso las coordenadas raw para convertirlas en
//string tipo UCI
function transformCoordinates(coord){
  x=coord[0];
  y=coord[1];

  //Click está dentro del rango permitido
  if(  (x>=1 && x<= 450) && ( y>=1 && y<=450 ) ){
    //Inicializo variables de salida
    var c1="" //X
    var c2="" //Y

    //Normalizo para quitar desfase
    //x=x-20;
    //y=y-20;

    //Obtengo el número del cuadrante del tablero
    cuadX = Math.ceil( x/56.25 );
    cuadY = Math.ceil( y/56.25 );

    //transformar cuadX a string
    switch(cuadX){
      case 1:
        c1="a";
        break;
      case 2:
        c1="b";
        break;
      case 3:
        c1="c";
        break;
      case 4:
        c1="d";
        break;
      case 5:
        c1="e";
        break;
      case 6:
        c1="f";
        break;
      case 7:
        c1="g";
        break;
      case 8:
        c1="h";
        break;
    }

    //transformar cuadY
    switch(cuadY){
      case 1:
        c2=8;
        break;
      case 2:
        c2=7;
        break;
      case 3:
        c2=6;
        break;
      case 4:
        c2=5;
        break;
      case 5:
        c2=4;
        break;
      case 6:
        c2=3;
        break;
      case 7:
        c2=2;
        break;
      case 8:
        c2=1;
        break;
    }

    //Concateno las dos partes del movimiento
    resultado =c1+String(c2);

    //Agrego el recuadro en el arreglo
    clicks.push(resultado);

    //Elimino duplicados
    clicks = [...new Set(clicks)];

    if( clicks.length==1  ){
      $.get("sel_piece", {x:cuadX-1,y:c2-1} , function(data){
        //Convierte el string en objeto
        var obj_svg=$(data)
        //Acceder a la info de la imagen
        var img_svg=obj_svg[0]
        dibujarSVG(img_svg)
      });
    }

    game()

  } else{
    console.log("Nok");
  }
}

async function getInfoHuman(click){
  console.log("Entro getInfoHuman")
  var info="";

  if( click.length>2 ){
   var move = click[0]+click[1]+click[2]
  } else {
    var move= click[0]+click[1]
  }

  console.log(move)

  
  info= await $.get("player",{player:"Humano",clicks:move,depth:0,game_id:game_id },function(data){
    //Data[0] viene info sobre la imagen SVG
    //Data[1] es la variable bool que indica si continua el juego

    if (data == "Movimiento invalido" ){
      console.log("Movimiento invalido getInfoHuman")
      return "Movimiento invalido"
    } else if ( data == "Promoción" ){
      console.log("Promoción")
      return "Promoción"
    }else{
    console.log("Correcto")
    //Convierte el string en objeto
    var obj_svg=$(data[0])
    //Acceder a la info de la imagen
    var img_svg=obj_svg[0]

    game_stats(data);

    return {cont:data[1],img:data[0]};

    }
    

  }).then(d =>d);

  return info;
}


//Obtengo las coordenadas raw
function getCoordinates(){
  var pageX = d3.event.pageX;
  var pageY = d3.event.pageY;

  var row_svg = document.getElementById("row-svg")
  var style_row = row_svg.currentStyle || window.getComputedStyle(row_svg)

  var col_svg = document.getElementById("col-svg")
  var style_col = col_svg.currentStyle || window.getComputedStyle(col_svg)

  x_norm = pageX - 40- Math.ceil(parseInt(style_col.marginLeft.slice(0,-2)))
  y_norm = pageY - 507
  
  /*
  console.log("------------------------------")

  console.log("Page X(Left): "+String(pageX )  )
  console.log("Col margin left " + String( Math.ceil(parseInt(style_col.marginLeft.slice(0,-2)))  ) )
  console.log("X normalized: " + String( x_norm )  )
  console.log("Cudrante X: " + String( Math.ceil( x_norm /56.25)  ) )

  console.log("---")
  console.log("Page Y(Top): "+String(pageY)  )
  console.log("This Offset Top: "+String(this.offsetTop)  )
  console.log("Col margin top " +  String(  style_col.marginTop ) )


  //console.log("Row margin left " + style_row.marginLeft )
  //console.log("Offsetleft row: " + String(row_svg.offsetLeft) )
  //console.log("Row margin+border+padding: " + String( row_svg.marginLeft) )
  //console.log("Offset left col: " +String(col_svg.offsetLeft) )
  //console.log("Page Y(Top): "+String(pageY)  )
  //console.log("This Offset Left: "+String(this.offsetLeft )  )
  //console.log("This Offset Top: "+String(this.offsetTop)  )

  console.log("------------------------------")
  */

  //LINEA DE PRUEBA
  transformCoordinates( [x_norm , y_norm ]  );
}

function renderClick(){
  console.log("Render Click")
  position = parseInt( this.id.split("-")[2] )
  console.log(position)
  switch(position){
    case 0:
      pieza = "q"
      break;
    case 1:
      pieza = "r"
      break;
    case 2:
      pieza = "n"
      break;
    default:
      pieza = "b"
      break;
  }

  /*
  if (color == false){
    pieza = pieza.toLowerCase()
  }
  */
  
  var row = document.getElementById("row-render")
  row.innerHTML="";

  clicks.push(pieza)
  game()

}

async function renderPieces(color){
  console.log("Entro renderPieces")
  console.log("Color actual: " +String(color) )
  var info="";
  info= await $.get("render_pieces",{color:color },function(data){
    //Data[0] viene info sobre la imagen SVG
    //Data[1] es la variable bool que indica si continua el juego

    var row = d3.select("#container-game-1").append("div").attr("class","row")
      .attr("id","row-render")
    
    var container = row.append("div").attr("class","col-xs-4 col-xs-offset-2")

      for ( i=0; i<data.length;i++ ){
        var columna = container.append("div").attr("class", "col-xs-3")
          .attr("id", "col-render-"+String(i) )
        var div = columna.append("div").attr("class","svg-render")
          .attr("id","svg-render-"+String(i))
        var svg = document.getElementById("svg-render-"+String(i))
        svg.innerHTML=""
        img = $(data[i])
        console.log(img)
        svg.appendChild( img[0] )

        var event_list= document.getElementById("col-render-"+String(i))
        event_list.addEventListener("click",renderClick,false );

      }
    
    var column


  }).then(d =>d);

  return info;
}



/////////////////////////////////////////////////////////////
async function game(){


  //Extrae el valor de la lista de seleccion del jugador en turno
  var tipo_jugador=getSelPlayer()

  if (continuar == true){
    if ( tipo_jugador=="Humano"  ){
      if( clicks.length>1){
        var data = await getInfoHuman(clicks)

        //Checo si la información que extraje es valida
        if ( data == "Movimiento invalido" ){
          clicks=[];
        } else if (data == "Promoción") {
          renderPieces(color);
        }else{
          dibujarSVG(data[0]);
          continuar = data[1];
          color = !color;
          clicks=[];
          game();
        }

      }   
    } else{
      var tipo_jug_serv = serverSelPlayer(tipo_jugador)
      var data = await getInfo(tipo_jug_serv, getSelDepth() )
      dibujarSVG(data[0]);
      continuar = data[1];
      color = !color;
      game();
    }
  }
}

//Obtener tablero SVG actual
function handleSubmit() {
  $.get("game",function(data){
   dibujarSVG(data)
  });
}

async function handlePlay(){
  var tipo_jugador=getSelPlayer()
  var tipo_jug_serv = serverSelPlayer(tipo_jugador)
  var data = await getInfo(tipo_jug_serv, getSelDepth() )
  dibujarSVG(data[0]);
  continuar = data[1];
  color = !color;
}

function save(){
  //Obtengo svg
  var svg = document.getElementById("svg").innerHTML 

  $.get("save",{svg:svg,game_id:game_id},function(data){
  })
  ////////////////////////////////////
  // Modificar HTML para mostrar mensaje que se guardó el juego de forma exitosa
  //Seleccionar el contenedor
  var contenedor = d3.select("#contenedor-text");
  //Quitar todos los elementos del contenedor
  contenedor.selectAll("*").remove();
  //Agregar titulo
  var h2s = contenedor.append("h1").attr("class","titulo")
  h2s.text("Successfully Saved Game")
}

initialize();
handleSubmit()

// Add event listener for submit button
d3.select("#submit").on("click", handleSubmit);

//Add event listener for submit button
d3.select("#play").on("click", handlePlay);

//Add event listener for submit button
d3.select("#game").on("click", game);

//Add event listener for Restart button
d3.select("#restart").on("click", initialize);

//Add event listener to click on board
d3.select("#svg").on("click",getCoordinates)

//Add event listener to click on board
d3.select("#save").on("click",save)