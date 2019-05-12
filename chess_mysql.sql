-- Borrar BD
-- DROP DATABASE IF EXISTS chess_db;

-- create database
-- CREATE DATABASE chess_db;

use chess_db;

create table plays(
	id int auto_increment NOT NULL,
    game_id int,
    mov_len int,
    end_game boolean,
    game_len int,
    x_dest int,
    y_dest int,
    playerplays_type varcharacter(2),
    depth int,
    winner varcharacter(10),
    end_game_mode varcharacter(22),
    exec_time decimal(10,6),
    score decimal(9,3),
    primary key(id)
);

Select max(id) from plays;

Select * from plays LIMIT 50000;

SELECT * FROM plays ORDER BY id DESC LIMIT 50000;

Select sum(mov_len) from plays;

-- DELETE FROM plays Where id>49272;

-- ALTER TABLE plays ADD COLUMN game_id int AFTER id;
-- ALTER TABLE plays ADD COLUMN score decimal(7,3) AFTER exec_time;
-- ALTER TABLE plays CHANGE score score decimal(9,3);
-- ALTER TABLE plays DROP COLUMN exec_time;
-- ALTER TABLE plays CHANGE exec_time exec_time decimal(9,5);
-- ALTER TABLE plays CHANGE y_axis y_dest int;

Select mov_len from plays order by mov_len desc;

Select game_len from plays order by game_len desc;

Select avg(game_len) from plays;

Select * from plays where winner<>'';

-- Test Query
Select end_game_mode,count(playerplays_type) 
from plays 
where winner<>'' and game_len>1 and game_len<633 and depth=0 and playerplays_type="M2"
group by end_game_mode;

Select max(game_id) from plays;
