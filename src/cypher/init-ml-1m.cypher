LOAD CSV FROM 'file:///movies.dat' AS row FIELDTERMINATOR '\t' CREATE (:Movie { MovieID:toInteger(row[0]), Title:row[1], Genres:row[2] });

CREATE INDEX MovieIndex FOR (m:Movie) ON (m.MovieID);

LOAD CSV FROM 'file:///users.dat' AS row FIELDTERMINATOR '\t' CREATE (:User { UserID:toInteger(row[0]), Gender:row[1], Age:toInteger(row[2]), Occupation:toInteger(row[3]), ZipCode:row[4] });

CREATE INDEX UserIndex FOR (u:User) ON (u.UserID);

:auto LOAD CSV FROM 'file:///ratings.dat' AS row FIELDTERMINATOR '\t' CALL { WITH row MATCH (u:User{UserID:toInteger(row[0])}) WITH row, u MATCH (m:Movie{MovieID:toInteger(row[1])}) WITH row, u, m CREATE (u)-[:Rate{ Rating:toInteger(row[2]), Timestamp:toInteger(row[3]) }]->(m) } IN TRANSACTIONS OF 500 rows;
