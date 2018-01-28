\include "event-romer.ly"
\paper {
  #(set-paper-size "letter")
}
\score {
  {
    \clef treble
    \key f \major
    \numericTimeSignature
    \time 4/4
    f'4 g' a' f'
    f' g' a' f'
    \mark "*"
    a'4 bes' c''2
    a'4 bes' c''2
    \mark "*"
    c''8 d'' c'' bes' a'4 f'
    c''8 d'' c'' bes' a'4 f'
    \mark "*"
    f'4 c' f'2
    f'4 c' f'2
  }
  \layout { }
  \midi { }
  \version "2.18.2"
}
