from flask import (Flask, render_template, redirect, Markup, request , 
    Response, jsonify,url_for)
import pymongo, json
import chess_scrape
import chess

from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()

import pandas as pd

engine = create_engine ("mysql://root:Aa1$0110m@localhost/chess_db")

from chess_engine import ( boardSVGRepr, initialize_game, 
    call_jugador_v4, global_board, global_turn, process_play, jugador_v1, jugador_v2,
    jugador_v3, jugador_v4, jugador_v5, get_move, fen_representation, initialize_game_fen,
    jugador_v6)


# Create an instance of Flask
app = Flask(__name__)  

#Setup mongo connection
conn = "mongodb://localhost:27017"
client = pymongo.MongoClient(conn)
db = client.chess

#Global Variables


global flagLoadGame
flagLoadGame=False
global globalFEN
globalFEN=""
global game_id
game_id=""

# Route to render index.html template using data from Mongo
@app.route("/")
def index():

    # Find one record of data from the mongo database
    # @TODO: YOUR CODE HERE!

    # Return template and data
    print("Home")

    number_of_games = len( list(db.games.find()))

    return render_template("index.html", numb_games= number_of_games)

@app.route("/load")
def load():

    print("Load")

    return render_template("load_page.html")

@app.route("/saved_games",methods=["GET"])
def saved_games():

    #Obtengo lista de todos los juegos guardados
    #Quito de la lsita de resultados el id ya que no es un objeto primitivo
    mongo_games = list( db.games.find({},{"_id":0}))

    return jsonify(mongo_games)

@app.route("/load_game",methods=["GET"])
def load_game():
    global flagLoadGame
    global globalFEN
    global game_id
    globalFEN = request.args.get("fen")
    flagLoadGame=True
    game_id = request.args.get("game_id")
    print(globalFEN)
    return "OK"


@app.route("/svg_test")
def test():
    return render_template("test_svg.html" )

@app.route("/statistics")
def statistics():
    return render_template("statistics_page.html" )

@app.route("/game", methods=["GET"])
def game():
    print("game")  
    #return Markup(boardSVGRepr(global_board()))
    return jsonify( boardSVGRepr(global_board())) 

@app.route("/sel_piece", methods=["GET"])
def sel_piece():
    x = int( request.args.get("x") )
    y = int( request.args.get("y") )
    arrows = [(chess.square(x,y), (chess.square(x,y)) )]
    temp_board = chess.svg.board(board= global_board(), arrows = arrows )
    return jsonify( boardSVGRepr( temp_board ) )



@app.route("/initialize", methods=["GET"])
def initialize():
    global flagLoadGame
    global globalFEN
    global game_id

    if  flagLoadGame==False:
        initialize_game()
        conn = engine.connect()
        max_game_id = pd.read_sql("Select max(game_id) from plays", conn)
        max_game_id = max_game_id.iloc[0][0]
        if max_game_id is None:
            max_game_id=-1
        new_game_id= max_game_id+1
        print(new_game_id)
    else:
        initialize_game_fen(globalFEN)
        flagLoadGame=False
        globalFEN=""
        text="FEN Initialize"
        new_game_id = game_id


    print("Game initialized")
    return str(new_game_id)

@app.route("/play",methods=["GET"]) 
def play_game():
    print("Play") 
    board = global_board()
    values = process_play( jugador_v4(board,color,0),"M4",0 )
    return jsonify(values) 

@app.route("/scrape",methods=["GET"])
def scrape():
    info = chess_scrape.scrape()
    return jsonify(info)

@app.route("/save", methods=["GET"])
def save():

    #Recibo string que representa imagen svg
    svg = request.args.get("svg")
    #Recibo game_id
    game_id = request.args.get("game_id")
    #Obtengo tablero actual
    board = global_board()
    #Obtengo representacion FEN del tablero
    fen = fen_representation(board)
    print("FEN:")
    print(fen)

    db.games.insert_one(
        {
            "svg":svg,
            "fen":fen,
            "game_id":game_id
        }
    )
    
    return jsonify("Info recibida")

@app.route("/mysql",methods=["GET"])
def mySQL():
    #Me conecto con la BD de MySQL
    print("Entro mysql")
    conn = engine.connect()
    plays_df = pd.read_sql("Select id,mov_len, playerplays_type,depth,game_len,winner FROM plays",conn)
    plays_df = plays_df.drop( ["id"] , axis=1)
    plays_dict = plays_df.to_dict("records")
    return jsonify(plays_dict)

@app.route("/heat_map", methods=["GET"])
def heat_map():
    conn = engine.connect()
    df = pd.read_sql("Select game_len, x_dest,y_dest,playerplays_type,depth FROM plays", conn)
    df_dict = df.to_dict("records")
    return jsonify(df_dict)

@app.route("/box_time", methods=["GET"])
def box_time():
    conn = engine.connect()
    df = pd.read_sql("Select game_len,playerplays_type,depth,exec_time FROM plays where exec_time<>'' ", conn)
    df_dict = df.to_dict("records")
    return jsonify(df_dict)

@app.route("/box_len", methods=["GET"])
def box_len():
    conn = engine.connect()
    df = pd.read_sql("Select game_len,playerplays_type,depth,mov_len FROM plays ", conn)
    df_dict = df.to_dict("records")
    return jsonify(df_dict)


@app.route("/bars",methods=["GET"])
def bars():
    conn = engine.connect()
    min_turn = request.args.get("min")
    max_turn = request.args.get("max")
    depth = request.args.get("depth")
    player = request.args.get("player")
    q1= "Select end_game_mode,count(playerplays_type) from plays "
    q2= "where winner<>'' "
    limit_turn = "and game_len>"+str(min_turn)+" and game_len<"+str(max_turn)+" "
    q4="group by end_game_mode"

    if(depth =="Depth 0"):
        depth_str = "and depth=0 "
    elif (depth == "Depth 1"):
        depth_str = "and depth=1 "
    elif (depth == "Depth 2"):
        depth_str = "and depth=2 "
    else:
        depth_str = ""

    
    machine_str ="and playerplays_type = 'M1' "
    str1= limit_turn + depth_str + machine_str
    df1= pd.read_sql( q1+q2+str1+q4, conn )
    labels1= list(df1["end_game_mode"])
    values1 = list( df1["count(playerplays_type)"] )
    info1=[labels1,values1]

    
    machine_str ="and playerplays_type = 'M2' "
    str2= limit_turn + depth_str + machine_str
    df2= pd.read_sql( q1+q2+str2+q4, conn )
    labels2= list(df2["end_game_mode"])
    values2 = list( df2["count(playerplays_type)"] )
    info2=[labels2,values2]

    
    machine_str ="and playerplays_type = 'M3' "
    str3= limit_turn + depth_str + machine_str
    df3= pd.read_sql( q1+q2+str3+q4, conn )
    labels3= list(df3["end_game_mode"])
    values3 = list( df3["count(playerplays_type)"] )
    info3=[labels3,values3]

    
    machine_str ="and playerplays_type = 'M4' "
    str4= limit_turn + depth_str + machine_str
    df4= pd.read_sql( q1+q2+str4+q4, conn )
    labels4= list(df4["end_game_mode"])
    values4 = list( df4["count(playerplays_type)"] )
    info4=[labels4,values4]


    machine_str ="and playerplays_type = 'M5' "
    str5= limit_turn + depth_str + machine_str
    df5= pd.read_sql( q1+q2+str5+q4, conn )
    labels5= list(df5["end_game_mode"])
    values5 = list( df5["count(playerplays_type)"] )
    info5=[labels5,values5]


    machine_str="and playerplays_type = 'M6' "
    str6= limit_turn + depth_str + machine_str
    df6= pd.read_sql( q1+q2+str6+q4, conn )
    labels6= list(df6["end_game_mode"])
    values6 = list( df6["count(playerplays_type)"] )
    info6=[labels6,values6]

    machine_str="and playerplays_type = 'H' "
    str7= limit_turn + depth_str + machine_str
    df7= pd.read_sql( q1+q2+str7+q4, conn )
    labels7= list(df7["end_game_mode"])
    values7 = list( df7["count(playerplays_type)"] )
    info7=[labels7,values7]

    
    return jsonify( [info1,info2,info3,info4,info5,info6,info7]  )



@app.route("/quickStats",methods=["GET"])
def quickStats():
    conn = engine.connect()
    df = pd.read_sql("Select * FROM plays",conn)
    #Depuro DF
    df = df.drop( ["x_dest","y_dest"] , axis=1)
    #DF que sólo contiene la info de las partidas acabadas
    tabla_fin_partida = df.loc[ df["winner"] != ""  ]
    total_movimientos= df["id"].max()
    total_posiciones_evaluadas = df["mov_len"].sum()
    total_de_partidas = tabla_fin_partida["winner"].count()
    #Agrupo tabla fin partida para id mejor y peor jugador
    temp = tabla_fin_partida.loc[ tabla_fin_partida["end_game_mode"] == "checkmate" ]
    clasif_jugadores = temp.groupby(["playerplays_type"])["winner"].count()
    #Mejor jugador
    mejor_jugador_name = clasif_jugadores.index[4]
    mejor_jugador_score = clasif_jugadores[4]
    #Peor jugador
    peor_jugador_name = clasif_jugadores.index[0]
    peor_jugador_score = clasif_jugadores[0]
    #Resultados
    info = [ int(total_movimientos), int(total_posiciones_evaluadas), int(total_de_partidas),  
        mejor_jugador_name, int(mejor_jugador_score), peor_jugador_name, int(peor_jugador_score)  ]
    return jsonify( info  )

@app.route("/render_pieces",methods=["GET"] )
def render_pieces():
    color = request.args.get("color")
    print(f"El color recibido es: {color} ")
    if color== "true" :
        svg = [ chess.svg.piece(chess.Piece.from_symbol("Q")),chess.svg.piece(chess.Piece.from_symbol("R")),
            chess.svg.piece(chess.Piece.from_symbol("N")),chess.svg.piece(chess.Piece.from_symbol("B"))]
    else:
        svg = [ chess.svg.piece(chess.Piece.from_symbol("q")),chess.svg.piece(chess.Piece.from_symbol("r")),
            chess.svg.piece(chess.Piece.from_symbol("n")),chess.svg.piece(chess.Piece.from_symbol("b"))]

    return jsonify(svg)



@app.route("/player", methods=["GET"])
def player():
    print("-----")
    #Obtener el tablero actual y turno
    board = global_board()
    color = global_turn()

    #Tipo de jugador
    #El valor depues de coma es el default
    player = request.args.get("player", "M1" )
    print(player)
    #Profundidad
    depth = int( request.args.get("depth",0))
    game_id = int(request.args.get("game_id"))
    print(depth)
    
    #Condicionales para saber quien atiende
    if player=="Humano":
        print("Humano")
        move = get_move( request.args.get("clicks") )

        if move == "Movimiento invalido":
            return "Movimiento invalido"
        elif move=="Promoción":
            return "Promoción"

        values = process_play( move , "H",0, game_id )
        print(".....")


    elif player =="M1":
        print("M1")
        values =  process_play( jugador_v1(board,color,depth) , "M1",0, game_id )
        

    elif player =="M2":
        print("M2")
        values =  process_play( jugador_v2(board,color,depth) , "M2",0, game_id )
    
    elif player =="M3":
        print("M3")
        values =  process_play( jugador_v3(board,color,depth) , "M3",0 , game_id )

    elif player =="M4":
        print("M4")
        values =  process_play( jugador_v4(board,color,depth) , "M4",0 , game_id )

    elif player =="M5":
        print("M5")
        values =  process_play( jugador_v5(board,color,depth) , "M5", depth , game_id  )
    elif player == "M6":
        print("M6")
        values = process_play( jugador_v6(board,color,depth)  ,"M6", depth , game_id )
    return jsonify(values)




if __name__ == "__main__":
    app.run(debug=True)
