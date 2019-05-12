import chess
import pandas as pd
import random
import multiprocessing as mp

wb_horse = pd.DataFrame({"a":[-5.0,-4.0,-3.0,-3.0,-3.0,-3.0,-4.0,-5.0],
                         "b":[-4.0,-2.0,00.0,00.5,00.0,00.5,-2.0,-4.0],
                         "c":[-3.0,00.0,01.0,01.5,01.5,01.0,00.0,-3.0],
                         "d":[-3.0,00.0,01.5,02.0,02.0,01.5,00.5,-3.0],
                         "e":[-3.0,00.0,01.5,02.0,02.0,01.5,00.5,-3.0],
                         "f":[-3.0,00.0,01.0,01.5,01.5,01.0,00.0,-3.0],
                         "g":[-4.0,-2.0,00.0,00.5,00.0,00.5,-2.0,-4.0],
                         "h":[-5.0,-4.0,-3.0,-3.0,-3.0,-3.0,-4.0,-5.0]})

wb_alfil = pd.DataFrame({"a":[-2.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0],
                         "b":[-1.0,00.0,00.0,00.5,00.0,01.0,00.5,-1.0],
                         "c":[-1.0,00.0,00.5,00.5,01.0,01.0,00.0,-1.0],
                         "d":[-1.0,00.0,01.0,01.0,01.0,01.0,00.0,-1.0],
                         "e":[-1.0,00.0,01.0,01.0,01.0,01.0,00.0,-1.0],
                         "f":[-1.0,00.0,00.5,00.5,01.0,01.0,00.0,-1.0],
                         "g":[-1.0,00.0,00.0,00.5,00.0,01.0,00.5,-1.0],
                         "h":[-2.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-2.0]})

w_rey = pd.DataFrame({"a":[-3.0,-3.0,-3.0,-3.0,-2.0,-1.0,02.0,02.0],
                      "b":[-4.0,-4.0,-4.0,-4.0,-3.0,-2.0,02.0,03.0],
                      "c":[-4.0,-4.0,-4.0,-4.0,-3.0,-2.0,-1.0,-1.0],
                      "d":[-5.0,-5.0,-5.0,-5.0,-4.0,-2.0,-1.0,-1.0],
                      "e":[-5.0,-5.0,-5.0,-5.0,-4.0,-2.0,-1.0,00.0],
                      "f":[-4.0,-4.0,-4.0,-4.0,-3.0,-2.0,-1.0,-1.0],
                      "g":[-4.0,-4.0,-4.0,-4.0,-3.0,-2.0,02.0,03.0],
                      "h":[-3.0,-3.0,-3.0,-3.0,-2.0,-1.0,02.0,02.0]})

b_rey = w_rey.iloc[::-1]

wb_queen = pd.DataFrame({"a":[-2.0,-1.0,-1.0,-0.5,00.0,-1.0,-1.0,-2.0],
                         "b":[-1.0,00.0,00.0,00.0,00.0,00.5,00.0,-1.0],
                         "c":[-1.0,00.0,00.5,00.5,00.5,00.5,00.5,-1.0],
                         "d":[-0.5,00.0,00.5,00.5,00.5,00.5,00.0,-0.5],
                         "e":[-0.5,00.0,00.5,00.5,00.5,00.5,00.0,-0.5],
                         "f":[-1.0,00.0,00.5,00.5,00.5,00.5,00.5,-1.0],
                         "g":[-1.0,00.0,00.0,00.0,00.0,00.5,00.0,-1.0],
                         "h":[-2.0,-1.0,-1.0,-0.5,00.0,-1.0,-1.0,-2.0]})

wb_torre = pd.DataFrame({"a":[00.0,00.5,-0.5,-0.5,-0.5,-0.5,-0.5,00.0],
                         "b":[00.0,01.0,00.0,00.0,00.0,00.0,00.0,00.0],
                         "c":[00.0,01.0,00.0,00.0,00.0,00.0,00.0,00.0],
                         "d":[00.0,01.0,00.0,00.0,00.0,00.0,00.0,00.5],
                         "e":[00.0,01.0,00.0,00.0,00.0,00.0,00.0,00.5],
                         "f":[00.0,01.0,00.0,00.0,00.0,00.0,00.0,00.0],
                         "g":[00.0,01.0,00.0,00.0,00.0,00.0,00.0,00.0],
                         "h":[00.0,00.5,-0.5,-0.5,-0.5,-0.5,-0.5,00.0]})

w_pawn = pd.DataFrame({"a":[10.0,5.0,1.0,0.5,0.0,00.5,0.0,0.0],
                       "b":[10.0,5.0,1.0,0.5,0.0,-0.5,0.0,0.0],
                       "c":[10.0,5.0,2.0,1.0,0.0,-1.0,0.0,0.0],
                       "d":[10.0,5.0,3.0,2.5,2.0,00.0,0.0,0.0],
                       "e":[10.0,5.0,3.0,2.5,2.0,00.0,0.0,0.0],
                       "f":[10.0,5.0,2.0,1.0,0.0,-1.0,0.0,0.0],
                       "g":[10.0,5.0,1.0,0.5,0.0,-0.5,0.0,0.0],
                       "h":[10.0,5.0,1.0,0.5,0.0,00.5,0.0,0.0]})

b_pawn = w_pawn.iloc[::-1]

wb_other = pd.DataFrame({"a":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
                         "b":[1.0,2.0,2.0,2.0,2.0,2.0,2.0,1.0],
                         "c":[1.0,2.0,3.0,3.0,3.0,3.0,2.0,1.0],
                         "d":[1.0,2.0,3.0,4.0,4.0,3.0,2.0,1.0],
                         "e":[1.0,2.0,3.0,4.0,4.0,3.0,2.0,1.0],
                         "f":[1.0,2.0,3.0,3.0,3.0,3.0,2.0,1.0],
                         "g":[1.0,2.0,2.0,2.0,2.0,2.0,2.0,1.0],
                         "h":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]})
  


def work_log(work_data):
    if work_data[0] == "A":
        return work_data[1]
    elif work_data[0] == "B":
        return work_data[1]**2
    elif work_data[0] == "C":
        return work_data[1]**3
    else:
        return work_data[1]**4

def analisis_v5(board,move,player_color, original_player_color):
    
    #Esta versión regresa positivo si juega blanco, negativo si juega negro
    
    ############################################ General ############################################
    
    #Puntaje inicial
    score= 0
    
    #Variable que ayuda al cálculo de score_is_attacked. Indica la proporción del valor de la pieza que se restará del puntaje
    proporcion_perder_pieza = 0.7
    
    #Diccionario para traducir letra a indice de la columna en el DF
    col_dict = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7}
    
    #Diccionario para los valores de las piezas
    dict_piezes_value ={"Pawn":10,"Bishop":30,"Knight":30,"Rook":50,"Queen":90,"King":0}
    
    #El movimiento viene formato UCI p.e.: "e3f3"
    # "e3" es el lugar de origen y "f3" es el destino
    move_str =  str(move)    
    #String que representa la posición de origen del movimiento
    move_org = move_str[0] + move_str[1] 
    #String que representa la posición de destino del movimiento
    move_des = move_str[2] + move_str[3]
    
    #Score para priorizar el movimiento de peones en el early game
    score_early_move=0
    #Score para priorizar el movimiento de peones en el late game
    score_late_move=0
    #Score inicial si estoy en jaque
    score_estoy_en_jaque=0
    #Valor inicial del score que obtengo si el movimiento provoca que me ponga en peligro de ser atacado
    score_is_attacked=0
    #Valor inicial del score que obtengo de la posición donde se vaya a mover la pieza
    score_mov=0
    #Score que da puntos si el movimiento resulta en comer las piezas del oponente
    score_tablero=0
    #Aumenta score si deja en jaque mate al oponente
    score_check_mate = 0
    #Penalizar que regrese a su posición anterior
    score_regreso=0
    #Dar puntos si pongo en jaque
    score_is_check=0
    #Dar puntos si, después del movimiento amenazo las piezas del oponente o cubro mis piezas
    score_is_attacking=0
    #Evitar empate si el tablero es favorable
    score_evitar_empate =0
    #Buscar empate si el tablero no es favorable
    score_buscar_empate=0
    #Score si estoy siendo atacado en mi casilla de origen
    score_origin_attacked=0
    #Score enroque
    score_is_castling=0
    
    ############################################ Mi Turno ############################################
    
    #Dar más puntos si se puede realizar el enroque
    score_is_castling = 10 if board.is_castling(move) else 0
    
    
    #Detecto si el movimiento de mi oponente me dejó en jaque
    estoy_en_jaque = board.is_check()
    
    #Detecto donde está la pieza antes de que se realice el movimiento
    pieza_org = str( board.piece_at( chess.SQUARES[chess.SQUARE_NAMES.index(move_org)] ))
    
    
    #Tabla para signar prioridad a que se mueva la pieza ya que la están atacando en su posición original
    if pieza_org == "P" or pieza_org == "p":
        score_origin_attacked = dict_piezes_value.get("Pawn")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_org)]))
    
    elif pieza_org =="B" or pieza_org == "b":
        score_origin_attacked = dict_piezes_value.get("Bishop")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_org)]))
        
    elif pieza_org=="R" or pieza_org=="r":
        score_origin_attacked = dict_piezes_value.get("Rook")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_org)]))
    
    elif pieza_org=="N" or pieza_org=="n":
        score_origin_attacked = dict_piezes_value.get("Knight")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_org)]))
        
    elif pieza_org=="Q" or pieza_org=="q":
        score_origin_attacked = dict_piezes_value.get("Queen")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_org)]))
    else:
        score_origin_attacked = dict_piezes_value.get("King")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_org)]))
    
    
    ############################################ Turno Oponente ############################################
    
    #Ejecutar movimiento
    board.push(move)
    
    #Pieza que se encuentra en el lugar de origen (e3)
    #Usando los metodos de la librería puedo pasar el string de la posición y obtener la pieza que se encuentra en ese lugar
    pieza_org = str( board.piece_at( chess.SQUARES[chess.SQUARE_NAMES.index(move_des)] ))
    
    
    #iloc[row,col]
    if pieza_org == "P":
        score_mov =  w_pawn.iloc[ int(move_des[1])-1 , col_dict.get(move_des[0]) ] 
        score_is_attacked = (-1)*dict_piezes_value.get("Pawn")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_des)]))
        #Aumento la puntuación del Peon si estuve en jaque (prefiero quitar el jaque moviendo el peon)
        score_estoy_en_jaque = (8 if estoy_en_jaque== True else 0)
        #Priorizar en el juego inicial tratar de balancear el movimiento de los peones
        score_early_move = 2 if len(board.move_stack) < 18 else 0
        #Priorizar en el juego tardio para tratar de balancear el movimiento de los peones
        score_late_move = 4 if len(board.move_stack)>60 else 0
        
        
    elif pieza_org == "p":
        score_mov =  b_pawn.iloc[ int(move_des[1])-1 , col_dict.get(move_des[0]) ] 
        score_is_attacked = (-1)*dict_piezes_value.get("Pawn")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_des)]))
        #Aumento la puntuación del Peon si estuve en jaque (prefiero quitar el jaque moviendo el peon)
        score_estoy_en_jaque = (8 if estoy_en_jaque== True else 0)
        #Priorizar en el juego inicial tratar de balancear el movimiento de los peones
        score_early_move = 2 if len(board.move_stack) < 18 else 0
        #Priorizar en el juego tardio para tratar de balancear el movimiento de los peones
        score_late_move = 4 if len(board.move_stack)>60 else 0
        
        
    elif pieza_org =="R" or pieza_org =="r":
        score_mov =  wb_torre.iloc[ int(move_des[1])-1 , col_dict.get(move_des[0]) ]
        score_is_attacked = (-1)*dict_piezes_value.get("Rook")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_des)]))
        
    elif pieza_org == "N" or pieza_org=="n":
        score_mov =  wb_horse.iloc[ int(move_des[1])-1 , col_dict.get(move_des[0]) ]
        score_is_attacked = (-1)*dict_piezes_value.get("Knight")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_des)]))
        
    elif pieza_org == "Q" or pieza_org=="q":
        score_mov =  wb_queen.iloc[ int(move_des[1])-1 , col_dict.get(move_des[0]) ]
        score_is_attacked = (-1)*dict_piezes_value.get("Queen")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_des)]))
        
    elif pieza_org == "B" or pieza_org=="b":
        score_mov =  wb_alfil.iloc[ int(move_des[1])-1 , col_dict.get(move_des[0]) ]
        score_is_attacked = (-1)*dict_piezes_value.get("Bishop")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_des)]))
        
    elif pieza_org == "K":
        score_mov =  w_rey.iloc[ int(move_des[1])-1 , col_dict.get(move_des[0]) ]
        score_is_attacked = (-1)*dict_piezes_value.get("King")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_des)]))        
        
    else:
        score_mov =  b_rey.iloc[ int(move_des[1])-1 , col_dict.get(move_des[0]) ]
        score_is_attacked = (-1)*dict_piezes_value.get("King")*(proporcion_perder_pieza) \
            *len(board.attackers(not player_color, chess.SQUARES[chess.SQUARE_NAMES.index(move_des)]))
    
    
    #ciclo que revisa en el tablero el par pieza/valor
    for (pieza,valor) in [(chess.PAWN,dict_piezes_value.get("Pawn")),(chess.BISHOP,dict_piezes_value.get("Bishop")),
                          (chess.KNIGHT,dict_piezes_value.get("Knight")),(chess.ROOK,dict_piezes_value.get("Rook")),
                          (chess.QUEEN,dict_piezes_value.get("Queen")),(chess.KING,dict_piezes_value.get("King"))]:
        #el puntaje será mayor a cero si (suponiendo misma cantidad de piezas blancas y negras) el movimiento resulta en
        #la captura de una pieza
        score_tablero += len(board.pieces(pieza,player_color))*valor
        score_tablero -= len(board.pieces(pieza,not player_color))*valor
    
    
    score_check_mate = 1000 if board.is_checkmate() else 0

    
    mov_anterior="    "
    if len(board.move_stack)>2:
        move_stack=list(board.move_stack)
        mov_anterior = str( move_stack[len(move_stack)-3])
        if move_str == (mov_anterior[2]+mov_anterior[3]+mov_anterior[0]+mov_anterior[1]):
            score_regreso+= -8.5
            
    if len(board.move_stack)>4:
        move_stack=list(board.move_stack)
        mov_anterior = str( move_stack[len(move_stack)-5])
        if move_str == (mov_anterior[2]+mov_anterior[3]+mov_anterior[0]+mov_anterior[1]):
            score_regreso+= -4
    
    if len(board.move_stack)>6:
        move_stack=list(board.move_stack)
        mov_anterior = str( move_stack[len(move_stack)-7])
        if move_str == (mov_anterior[2]+mov_anterior[3]+mov_anterior[0]+mov_anterior[1]):
            score_regreso+= -4

    
    score_is_check = 3 if board.is_check() else 0
    
    
    #Lista de lugares donde puede moverse desde su posición actual
    list_possible_squares = list(board.attacks(chess.SQUARES[chess.SQUARE_NAMES.index(move_des)]))
    #Checar si esos cuadros están vacíos o no
    #Considera piezas amigas como enemigas
    list_attacked_pieces = [square for square in list_possible_squares if  
                            str(board.piece_at(square)).upper() in ["P","B","N","K","Q","R"] ]
    score_is_attacking = (len(list_attacked_pieces)*1) * (0.4 if pieza_org.upper()=="Q" else 1 ) *(0.75 if pieza_org.upper()=="N" else 1)
    
    
    #Le bajo la puntación al movimiento ya que trato de evitar un empate
    if (board.is_stalemate() or board.is_fivefold_repetition() or board.is_insufficient_material() or board.can_claim_draw()) and \
        score_tablero>=0 :
        score_evitar_empate= -1000
    
    
    #Le aumento la puntuación al tablero ya que voy perdiendo y busco al menos empatar
    if (board.is_stalemate() or board.is_fivefold_repetition() or board.is_insufficient_material() or board.can_claim_draw()) and \
        score_tablero<(((dict_piezes_value.get("Knight")+dict_piezes_value.get("Rook")+dict_piezes_value.get("Bishop"))*(-2)) - dict_piezes_value.get("Queen")) :
        score_buscar_empate= 500
    

    #El score final es el resultado de todos los scores incluyendo un valor de random
    score = score_mov + score_tablero + score_regreso + score_check_mate + score_is_check + score_is_attacked \
            + score_is_attacking + score_is_castling + score_estoy_en_jaque + score_late_move + score_early_move \
            + score_evitar_empate + score_buscar_empate + score_origin_attacked #+ random.random()/10
    
    #def analisis_v5(board,move,player_color, original_player_color):
    
#     if player_color == original_player_color:
#         return score
#     else:
#         if score<0:
#             return score
#         else:
#             return -score
    
    if player_color == original_player_color:
        return score
    else:
        return -score


def minimax ( depth , board , alpha , beta , is_maximizing, move, color_jugador ):
    if ( depth==0 ):
        #analisis_v5(board,move,player_color):
        board.pop()
        return analisis_v5( board , move, board.turn , color_jugador )
    
    possibleMoves = list(board.legal_moves)
    
    ################ Inicia Ordenamiento
    lista_ord=[]
    list_tupple=[]

    for move in possibleMoves:
        move_str = str(move)
        move_org = move_str[0]+move_str[1]
        pieza_org = str( board.piece_at( chess.SQUARES[chess.SQUARE_NAMES.index(move_org)] )).upper()
        dict_piezes_value ={"P":10,"B":30,"N":30,"R":50,"Q":90,"K":0}
        tupple = [move, dict_piezes_value[pieza_org]]
        list_tupple.append(tupple)

    list_tupple.sort( reverse=True , key = lambda x:x[1])
    for tup in list_tupple:
        lista_ord.append( tup[0] )

    possibleMoves = lista_ord
    
    
    ################ Finaliza Ordenamiento
    
    if(is_maximizing):
        bestMove = -99999
        
        for x in possibleMoves:
            move = x
            board.push(move)
            bestMove = max( bestMove , minimax(depth-1 , board , alpha , beta , not is_maximizing , move, color_jugador) )
            board.pop()
            alpha = max( alpha , bestMove )
            if beta <= alpha:
                return bestMove
        return bestMove
    else:
        bestMove = 99999
        for x in possibleMoves:
            move = x
            board.push(move)
            bestMove = min( bestMove , minimax( depth-1, board, alpha,beta, not is_maximizing , move, color_jugador ) )
            board.pop()
            beta = min(beta, bestMove)
            if(beta<=alpha):
                return bestMove
        return bestMove





def jugador_mp( info):
    board = info[0]
    board.push(info[1]) 
    color_jugador = board.turn
    depth = info[2]-1
    
    possibleMoves = list(board.legal_moves)
    bestMove = -99999
    bestMoveFinal = possibleMoves[0]
    
    ################ Inicia Ordenamiento
    lista_ord=[]
    list_tupple=[]

    for move in possibleMoves:
        move_str = str(move)
        move_org = move_str[0]+move_str[1]
        pieza_org = str( board.piece_at( chess.SQUARES[chess.SQUARE_NAMES.index(move_org)] )).upper()
        dict_piezes_value ={"P":10,"B":30,"N":30,"R":50,"Q":90,"K":0}
        tupple = [move, dict_piezes_value[pieza_org]]
        list_tupple.append(tupple)

    list_tupple.sort( reverse=True , key = lambda x:x[1])
    
    for tup in list_tupple:
        lista_ord.append( tup[0] )

    possibleMoves = lista_ord
    ################ Finaliza Ordenamiento
    
    for x in possibleMoves:
        move = x
        board.push(move)
        value = max( bestMove , minimax(depth-1,board,-100000,100000,  color_jugador , move, not color_jugador ) )
        board.pop()

        
        if( value > bestMove ):
            bestMove = value
            bestMoveFinal = move
    
    return [ info[1] , bestMove]


####################################
####################################
####################################
####################################
####################################

# def jugador_v7(board , color_jugador , depth):
#     lista_ord=[]
#     possibleMoves = list ( board.legal_moves )
    
#     global len1
#     global len2
    
#     if len(possibleMoves) == 1:
#         return possibleMoves[0].uci()
#     else:
#         depth=depth+1
        
#         #### Generar info que se va a pasar
#         lista_board=[]
#         print("Inicializa lista_board")
#         for x in possibleMoves:
#             temp =[ board.copy() , x , depth ]
#             print(board.copy())
#             print(x)
#             print("--")
#             lista_board.append(temp)
#         ####

#         pool = mp.Pool(8)
#         result_list =  pool.map( jugador_mp , lista_board )

#         result_list.sort( reverse=True , key = lambda x:x[1])
    
#         print("")
#         print("------------------")
#         print("Lista de resultados")
#         for tupple in result_list:
#             print(tupple[0])
#             print(tupple[1])
#             print("--")
#             lista_ord.append( tupple[0] )
            
#         len1=lista_board
#         len2=result_list
        
#         return lista_ord[0].uci()

# if __name__ == "__main__":
#     board = chess.Board()
#     jugador_v7(board , board.turn , 4)


