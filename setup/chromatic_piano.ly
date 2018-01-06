\include "event-listener.ly"
\paper {
  #(set-paper-size "letter")
}
\score {
  \new GrandStaff <<
    \new Staff {
      \numericTimeSignature
      \time 4/4
      \clef treble
      r2        r4  c'   cis'   d'  dis'  e'   f'  fis'   g'  gis'
      a'  ais'  b'  c''  cis''  d'' dis'' e''  f'' fis''  g'' gis''
      a'' ais'' b'' c''' cis'''
    }
    \new Staff {
      \numericTimeSignature
      \time 4/4
      \clef bass
      a,,  ais,,  b,,  c,  cis,  d, dis, e,  f, fis,  g, gis,
      a,   ais,   b,   c   cis   d  dis  e   f  fis   g  gis
      a    ais    b    c'  cis'
    }
  >>
  \layout { }
  \midi { }
  \version "2.18.2"
}
