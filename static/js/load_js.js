//Variable global
var globalFEN=[]

function initialize(){
    $.get("saved_games", function(data){

        for (i=0; i<data.length ; i++){

            //Agrego los FENs a la variable global
            globalFEN.push( data[i].fen )

            //Agrego las secciones necesarias usando D3
            console.log(data[i].fen.split(" "))
            var paneles = d3.select("#paneles").append("div").attr("class","panel panel-default")
            var body = paneles.append("div").attr("class","panel-body")
                .attr("id","panel-body-"+String(i)+"-"+String(data[i].game_id))
            var rows = body.append("div").attr("class","row")

            //Sección para dibujar la imagen
            var cols1 = rows.append("div").attr("class","col-xs-5" )
            var divs = cols1.append("div").attr("id","svg"+ String(i) )
                .attr("class","svg-img")
            //Convierto a objeto el string del svg
            img = $(data[i].svg);
            //Obtengo el div que va a guardar las imagenes svg
            var svg = document.getElementById("svg"+ String(i))
            //Limpio si ya tienen info
            svg.innerHTML="";
            //Agrego imagen SVG
            svg.appendChild(img[0]);
            
            //Modifico el tamaño de la imagen
            var img_r = d3.selectAll("svg")
            img_r.attr("height",400)
            img_r.attr("width",400)

            //Escribir la info de texto
            var cols2 = rows.append("div").attr("class","col-xs-4 col-xs-offset-1")

            //Escribir H
            var h_text = cols2.append("h2").attr("class","h-paneles")
            h_text.text("Game Slot #"+ String(i+1) )
            var br0 = cols2.append("br").attr("class","p-br")

            //Escribir Ps
            var p1 = cols2.append("p").attr("class","p-label").text("FEN del juego:")
            var li1 = cols2.append("li").attr("class","p-li").text(data[i].fen)
            var br1 = cols2.append("br").attr("class","p-br")

            var p3 = cols2.append("p").attr("class","p-label").text("Jugador a mover:")
            var prox_jugador = data[i].fen.split(" ")[1]
            var li3 = cols2.append("li").attr("class","p-li").text(prox_jugador);
            var br3 = cols2.append("br").attr("class","p-br")

            var p2 = cols2.append("p").attr("class","p-label").text("Turno:")
            var prox_turno = data[i].fen.split(" ")[ data[i].fen.split(" ").length-1   ]
            if( prox_jugador=="b" ){
                prox_turno = String(  (parseInt( prox_turno )*2)-1 )
            } else{
                prox_turno= String(  (parseInt(prox_turno)*2)-2  )
            }
            var li2 = cols2.append("li").attr("class","p-li").text(prox_turno);
            var br2 = cols2.append("br").attr("class","p-br")

            var p4 = cols2.append("p").attr("class","p-label").text("Id del juego:")
            var li3 = cols2.append("li").attr("class","p-li").text(data[i].game_id)
            var br3 = cols2.append("br").attr("class","p-br")

        }

        //Una vez cargada la página obtengo una lista de todos los paneles
        var list_panels = document.getElementsByClassName("panel-body");

        //A todos los paneles les agrego un event listener
        for (var i = 0; i < list_panels.length; i++) {
            list_panels[i].addEventListener('click', panelClick, false);
        }

    });

}

function panelClick(){
    //Levantar bandera en servidor para indicar que hay juego cargado

    //Obtengo el FEN del panel sobre el cual se hizo click
    fen= globalFEN[ parseInt( this.id.split("-")[2]) ]
    game_id = this.id.split("-",)[3]
    console.log(this.id)
    console.log(game_id)

    $.get("load_game",{fen:fen,game_id:game_id},function(data){
        window.location.href = "/svg_test"
    });

}



initialize();



