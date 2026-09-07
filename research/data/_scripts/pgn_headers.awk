# Full-pass PGN header distributions.
# Usage:
#   LC_ALL=C grep -a '^\[[A-Za-z]' games.pgn | LC_ALL=C awk -f pgn_headers.awk
# The grep prefilter matters: feeding movetext through awk as well takes
# roughly four times as long for the same result.
{
  tag = substr($1, 2)
  v = $0; sub(/^\[[A-Za-z0-9_]+ "/, "", v); sub(/"\]$/, "", v)
  if      (tag == "Event")       { games++; ev[v]++ }
  else if (tag == "Result")      { res[v]++ }
  else if (tag == "WhiteElo" || tag == "BlackElo") {
    e = v + 0
    if (e > 0) {
      elosum += e; elon++
      if (elon == 1 || e < elomin) elomin = e
      if (e > elomax) elomax = e
      eb[int(e / 100) * 100]++
    }
  }
  else if (tag == "TimeControl") { tc[v]++ }
  else if (tag == "Termination") { term[v]++ }
  else if (tag == "ECO")         { eco[v]++ }
  else if (tag == "Opening")     { op[v]++ }
  else if (tag == "WhiteTitle" || tag == "BlackTitle") { ti[v]++ }
  else if (tag == "White" || tag == "Black")           { pl[v]++ }
  else if (tag == "UTCDate")     { dd[v]++ }
}
END {
  printf "GAMES\t%d\n", games
  printf "ELO\tn=%d mean=%.1f min=%d max=%d\n", elon, elosum / elon, elomin, elomax
  c = 0; for (k in pl)  c++; printf "UNIQUE_PLAYERS\t%d\n", c
  c = 0; for (k in eco) c++; printf "UNIQUE_ECO\t%d\n", c
  c = 0; for (k in op)  c++; printf "UNIQUE_OPENING\t%d\n", c
  print "--DATE--";        for (k in dd)   printf "%s\t%d\n", k, dd[k]
  print "--RESULT--";      for (k in res)  printf "%s\t%d\n", k, res[k]
  print "--EVENT--";       for (k in ev)   printf "%s\t%d\n", k, ev[k]
  print "--TERMINATION--"; for (k in term) printf "%s\t%d\n", k, term[k]
  print "--TITLE--";       for (k in ti)   printf "%s\t%d\n", k, ti[k]
  print "--ELOBUCKET--";   for (k in eb)   printf "%s\t%d\n", k, eb[k]
  print "--TIMECONTROL--"; for (k in tc)   printf "%s\t%d\n", k, tc[k]
  print "--OPENING--";     for (k in op)   printf "%s\t%d\n", k, op[k]
  print "--ECO--";         for (k in eco)  printf "%s\t%d\n", k, eco[k]
}
