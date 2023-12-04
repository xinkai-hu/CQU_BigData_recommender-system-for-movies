LOAD CSV FROM 'file:///movies.dat' AS row FIELDTERMINATOR '\t' CREATE (:Movie { MovieID:toInteger(row[0]), Title:row[1], Genres:row[2] });

CREATE INDEX MovieIndex FOR (m:Movie) ON (m.MovieID);

LOAD CSV FROM 'file:///users.dat' AS row FIELDTERMINATOR '\t' CREATE (:User { UserID:toInteger(row[0]) });

CREATE INDEX UserIndex FOR (u:User) ON (u.UserID);

:auto LOAD CSV FROM 'file:///ratings.dat' AS row FIELDTERMINATOR '\t' CALL { WITH row MATCH (u:User{UserID:toInteger(row[0])}) WITH row, u MATCH (m:Movie{MovieID:toInteger(row[1])}) WITH row, u, m CREATE (u)-[:Rate{ Rating:toInteger(row[2]), Timestamp:toInteger(row[3]) }]->(m) } IN TRANSACTIONS OF 500 rows;

:auto LOAD CSV FROM 'file:///tags.dat' AS row FIELDTERMINATOR '\t' CALL { WITH row MATCH (u:User{UserID:toInteger(row[0])}) WITH row, u MATCH (m:Movie{MovieID:toInteger(row[1])}) WITH row, u, m CREATE (u)-[:Comment{ Tag:row[2], Timestamp:toInteger(row[3]) }]->(m) } IN TRANSACTIONS OF 500 rows;
