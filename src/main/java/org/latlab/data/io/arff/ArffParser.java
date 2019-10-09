/* Generated By:JavaCC: Do not edit this line. ArffParser.java */
package org.latlab.data.io.arff;

import org.latlab.data.Instance;
import org.latlab.data.MixedDataSet;
import org.latlab.util.DiscreteVariable;
import org.latlab.util.SingularContinuousVariable;
import org.latlab.util.Variable;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

//import org.org.latlab.data.io.IntegerText;

@SuppressWarnings({ "unused", "serial" })
public class ArffParser implements ArffParserConstants {
        public static MixedDataSet parse(InputStream stream) throws ParseException {
                ArffParser parser = new ArffParser(stream, "UTF-8");
                return parser.Arff();
        }

  final public MixedDataSet Arff() throws ParseException {
        String name;
        Variable variable;
        List<Variable> variables = new ArrayList<Variable>();
        List<Instance> instances;
    name = Relation();
    label_1:
    while (true) {
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case ATTRIBUTE:
        ;
        break;
      default:
        jj_la1[0] = jj_gen;
        break label_1;
      }
      variable = Attribute();
                                   variables.add(variable);
    }
    instances = Data(variables);
    jj_consume_token(0);
          {if (true) return new MixedDataSet(name, variables, instances);}
    throw new Error("Missing return statement in function");
  }

  final public String Relation() throws ParseException {
        String name;
    jj_consume_token(RELATION);
    name = StringLiteral();
                                          {if (true) return name;}
    throw new Error("Missing return statement in function");
  }

  final public Variable Attribute() throws ParseException {
        Variable variable;
        String name;
        String state;
        List<String> states;
    jj_consume_token(ATTRIBUTE);
    name = StringLiteral();
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case 26:
      states = NominalSpecification();
                  variable = new DiscreteVariable(name, states);
      break;
    case REAL:
    case NUMERIC:
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case REAL:
        jj_consume_token(REAL);
        break;
      case NUMERIC:
        jj_consume_token(NUMERIC);
        break;
      default:
        jj_la1[1] = jj_gen;
        jj_consume_token(-1);
        throw new ParseException();
      }
                  variable = new SingularContinuousVariable(name);
      break;
    case INTEGER:
      jj_consume_token(INTEGER);
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case 23:
        jj_consume_token(23);
        jj_consume_token(INTEGER_LITERAL);
        jj_consume_token(24);
        jj_consume_token(INTEGER_LITERAL);
        jj_consume_token(25);
        break;
      default:
        jj_la1[2] = jj_gen;
        ;
      }
                  variable = new SingularContinuousVariable(name);
      break;
    default:
      jj_la1[3] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
          {if (true) return variable;}
    throw new Error("Missing return statement in function");
  }

  final public List<String> NominalSpecification() throws ParseException {
        List<String> states = new ArrayList<String>();
    jj_consume_token(26);
    NominalState(states);
    label_2:
    while (true) {
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case 24:
        ;
        break;
      default:
        jj_la1[4] = jj_gen;
        break label_2;
      }
      jj_consume_token(24);
      NominalState(states);
    }
    jj_consume_token(27);
                {if (true) return states;}
    throw new Error("Missing return statement in function");
  }

  final public void NominalState(List<String> states) throws ParseException {
        String state;
        Token t;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case REAL:
    case NUMERIC:
    case INTEGER:
    case DQUOTED_STRING_LITERAL:
    case SQUOTED_STRING_LITERAL:
    case STRING_LITERAL:
      state = StringLiteral();
                                states.add(state);
      break;
    case INTEGER_LITERAL:
      t = jj_consume_token(INTEGER_LITERAL);
                              states.add(t.image);
      break;
    default:
      jj_la1[5] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
  }

  final public List<Instance> Data(List<Variable> variables) throws ParseException {
        Instance instance;
        List<Instance> instances = new ArrayList<Instance>();
    jj_consume_token(DATA);
    label_3:
    while (true) {
      switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
      case REAL:
      case NUMERIC:
      case INTEGER:
      case INTEGER_LITERAL:
      case FLOAT_LITERAL:
      case DQUOTED_STRING_LITERAL:
      case SQUOTED_STRING_LITERAL:
      case STRING_LITERAL:
      case 28:
        ;
        break;
      default:
        jj_la1[6] = jj_gen;
        break label_3;
      }
      instance = Instance(variables);
                                               instances.add(instance);
    }
          {if (true) return instances;}
    throw new Error("Missing return statement in function");
  }

  final public Instance Instance(List<Variable> variables) throws ParseException {
        List<String> values = new ArrayList<String>(variables.size());
        double weight = 1;
    Value(values);
    label_4:
    while (true) {
      if (jj_2_1(2)) {
        ;
      } else {
        break label_4;
      }
      jj_consume_token(24);
      Value(values);
    }
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case 24:
      jj_consume_token(24);
      jj_consume_token(26);
      weight = FloatLiteral();
      jj_consume_token(27);
      break;
    default:
      jj_la1[7] = jj_gen;
      ;
    }
          {if (true) return Instance.create(variables, values, weight);}
    throw new Error("Missing return statement in function");
  }

  final public void Value(List<String> values) throws ParseException {
        String value;
        Token t;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case REAL:
    case NUMERIC:
    case INTEGER:
    case DQUOTED_STRING_LITERAL:
    case SQUOTED_STRING_LITERAL:
    case STRING_LITERAL:
      value = StringLiteral();
                                values.add(value);
      break;
    case INTEGER_LITERAL:
      t = jj_consume_token(INTEGER_LITERAL);
                              values.add(t.image);
      break;
    case FLOAT_LITERAL:
      t = jj_consume_token(FLOAT_LITERAL);
                            values.add(t.image);
      break;
    case 28:
      jj_consume_token(28);
              values.add(null);
      break;
    default:
      jj_la1[8] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
  }

  final public String StringLiteral() throws ParseException {
        Token t;
        String value;
    switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
    case DQUOTED_STRING_LITERAL:
      t = jj_consume_token(DQUOTED_STRING_LITERAL);
                          value = t.image.substring(1, t.image.length() -1);
      break;
    case SQUOTED_STRING_LITERAL:
      t = jj_consume_token(SQUOTED_STRING_LITERAL);
                          value = t.image.substring(1, t.image.length() -1);
      break;
    case STRING_LITERAL:
      t = jj_consume_token(STRING_LITERAL);
                          value = t.image;
      break;
    case REAL:
      t = jj_consume_token(REAL);
                          value = t.image;
      break;
    case NUMERIC:
      t = jj_consume_token(NUMERIC);
                          value = t.image;
      break;
    case INTEGER:
      t = jj_consume_token(INTEGER);
                 value = t.image;
      break;
    default:
      jj_la1[9] = jj_gen;
      jj_consume_token(-1);
      throw new ParseException();
    }
                {if (true) return value;}
    throw new Error("Missing return statement in function");
  }

  final public int IntegerLiteral() throws ParseException {
        Token t;
    t = jj_consume_token(INTEGER_LITERAL);
          {if (true) return Integer.parseInt(t.image);}
    throw new Error("Missing return statement in function");
  }

  final public double FloatLiteral() throws ParseException {
        Token t;
    t = jj_consume_token(FLOAT_LITERAL);
          {if (true) return Double.parseDouble(t.image);}
    throw new Error("Missing return statement in function");
  }

  private boolean jj_2_1(int xla) {
    jj_la = xla; jj_lastpos = jj_scanpos = token;
    try { return !jj_3_1(); }
    catch(LookaheadSuccess ls) { return true; }
    finally { jj_save(0, xla); }
  }

  private boolean jj_3R_7() {
    if (jj_scan_token(INTEGER_LITERAL)) return true;
    return false;
  }

  private boolean jj_3R_12() {
    if (jj_scan_token(SQUOTED_STRING_LITERAL)) return true;
    return false;
  }

  private boolean jj_3R_6() {
    if (jj_3R_10()) return true;
    return false;
  }

  private boolean jj_3R_5() {
    Token xsp;
    xsp = jj_scanpos;
    if (jj_3R_6()) {
    jj_scanpos = xsp;
    if (jj_3R_7()) {
    jj_scanpos = xsp;
    if (jj_3R_8()) {
    jj_scanpos = xsp;
    if (jj_3R_9()) return true;
    }
    }
    }
    return false;
  }

  private boolean jj_3R_11() {
    if (jj_scan_token(DQUOTED_STRING_LITERAL)) return true;
    return false;
  }

  private boolean jj_3R_10() {
    Token xsp;
    xsp = jj_scanpos;
    if (jj_3R_11()) {
    jj_scanpos = xsp;
    if (jj_3R_12()) {
    jj_scanpos = xsp;
    if (jj_3R_13()) {
    jj_scanpos = xsp;
    if (jj_3R_14()) {
    jj_scanpos = xsp;
    if (jj_3R_15()) {
    jj_scanpos = xsp;
    if (jj_3R_16()) return true;
    }
    }
    }
    }
    }
    return false;
  }

  private boolean jj_3R_16() {
    if (jj_scan_token(INTEGER)) return true;
    return false;
  }

  private boolean jj_3R_15() {
    if (jj_scan_token(NUMERIC)) return true;
    return false;
  }

  private boolean jj_3R_14() {
    if (jj_scan_token(REAL)) return true;
    return false;
  }

  private boolean jj_3R_9() {
    if (jj_scan_token(28)) return true;
    return false;
  }

  private boolean jj_3R_13() {
    if (jj_scan_token(STRING_LITERAL)) return true;
    return false;
  }

  private boolean jj_3_1() {
    if (jj_scan_token(24)) return true;
    if (jj_3R_5()) return true;
    return false;
  }

  private boolean jj_3R_8() {
    if (jj_scan_token(FLOAT_LITERAL)) return true;
    return false;
  }

  /** Generated Token Manager. */
  public ArffParserTokenManager token_source;
  SimpleCharStream jj_input_stream;
  /** Current token. */
  public Token token;
  /** Next token. */
  public Token jj_nt;
  private int jj_ntk;
  private Token jj_scanpos, jj_lastpos;
  private int jj_la;
  private int jj_gen;
  final private int[] jj_la1 = new int[10];
  static private int[] jj_la1_0;
  static {
      jj_la1_init_0();
   }
   private static void jj_la1_init_0() {
      jj_la1_0 = new int[] {0x1000,0xc000,0x800000,0x401c000,0x1000000,0x75c000,0x107dc000,0x1000000,0x107dc000,0x71c000,};
   }
  final private JJCalls[] jj_2_rtns = new JJCalls[1];
  private boolean jj_rescan = false;
  private int jj_gc = 0;

  /** Constructor with InputStream. */
  public ArffParser(InputStream stream) {
     this(stream, null);
  }
  /** Constructor with InputStream and supplied encoding */
  public ArffParser(InputStream stream, String encoding) {
    try { jj_input_stream = new SimpleCharStream(stream, encoding, 1, 1); } catch(java.io.UnsupportedEncodingException e) { throw new RuntimeException(e); }
    token_source = new ArffParserTokenManager(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 10; i++) jj_la1[i] = -1;
    for (int i = 0; i < jj_2_rtns.length; i++) jj_2_rtns[i] = new JJCalls();
  }

  /** Reinitialise. */
  public void ReInit(InputStream stream) {
     ReInit(stream, null);
  }
  /** Reinitialise. */
  public void ReInit(InputStream stream, String encoding) {
    try { jj_input_stream.ReInit(stream, encoding, 1, 1); } catch(java.io.UnsupportedEncodingException e) { throw new RuntimeException(e); }
    token_source.ReInit(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 10; i++) jj_la1[i] = -1;
    for (int i = 0; i < jj_2_rtns.length; i++) jj_2_rtns[i] = new JJCalls();
  }

  /** Constructor. */
  public ArffParser(java.io.Reader stream) {
    jj_input_stream = new SimpleCharStream(stream, 1, 1);
    token_source = new ArffParserTokenManager(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 10; i++) jj_la1[i] = -1;
    for (int i = 0; i < jj_2_rtns.length; i++) jj_2_rtns[i] = new JJCalls();
  }

  /** Reinitialise. */
  public void ReInit(java.io.Reader stream) {
    jj_input_stream.ReInit(stream, 1, 1);
    token_source.ReInit(jj_input_stream);
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 10; i++) jj_la1[i] = -1;
    for (int i = 0; i < jj_2_rtns.length; i++) jj_2_rtns[i] = new JJCalls();
  }

  /** Constructor with generated Token Manager. */
  public ArffParser(ArffParserTokenManager tm) {
    token_source = tm;
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 10; i++) jj_la1[i] = -1;
    for (int i = 0; i < jj_2_rtns.length; i++) jj_2_rtns[i] = new JJCalls();
  }

  /** Reinitialise. */
  public void ReInit(ArffParserTokenManager tm) {
    token_source = tm;
    token = new Token();
    jj_ntk = -1;
    jj_gen = 0;
    for (int i = 0; i < 10; i++) jj_la1[i] = -1;
    for (int i = 0; i < jj_2_rtns.length; i++) jj_2_rtns[i] = new JJCalls();
  }

  private Token jj_consume_token(int kind) throws ParseException {
    Token oldToken;
    if ((oldToken = token).next != null) token = token.next;
    else token = token.next = token_source.getNextToken();
    jj_ntk = -1;
    if (token.kind == kind) {
      jj_gen++;
      if (++jj_gc > 100) {
        jj_gc = 0;
        for (int i = 0; i < jj_2_rtns.length; i++) {
          JJCalls c = jj_2_rtns[i];
          while (c != null) {
            if (c.gen < jj_gen) c.first = null;
            c = c.next;
          }
        }
      }
      return token;
    }
    token = oldToken;
    jj_kind = kind;
    throw generateParseException();
  }

  static private final class LookaheadSuccess extends Error { }
  final private LookaheadSuccess jj_ls = new LookaheadSuccess();
  private boolean jj_scan_token(int kind) {
    if (jj_scanpos == jj_lastpos) {
      jj_la--;
      if (jj_scanpos.next == null) {
        jj_lastpos = jj_scanpos = jj_scanpos.next = token_source.getNextToken();
      } else {
        jj_lastpos = jj_scanpos = jj_scanpos.next;
      }
    } else {
      jj_scanpos = jj_scanpos.next;
    }
    if (jj_rescan) {
      int i = 0; Token tok = token;
      while (tok != null && tok != jj_scanpos) { i++; tok = tok.next; }
      if (tok != null) jj_add_error_token(kind, i);
    }
    if (jj_scanpos.kind != kind) return true;
    if (jj_la == 0 && jj_scanpos == jj_lastpos) throw jj_ls;
    return false;
  }


/** Get the next Token. */
  final public Token getNextToken() {
    if (token.next != null) token = token.next;
    else token = token.next = token_source.getNextToken();
    jj_ntk = -1;
    jj_gen++;
    return token;
  }

/** Get the specific Token. */
  final public Token getToken(int index) {
    Token t = token;
    for (int i = 0; i < index; i++) {
      if (t.next != null) t = t.next;
      else t = t.next = token_source.getNextToken();
    }
    return t;
  }

  private int jj_ntk() {
    if ((jj_nt=token.next) == null)
      return (jj_ntk = (token.next=token_source.getNextToken()).kind);
    else
      return (jj_ntk = jj_nt.kind);
  }

  private List<int[]> jj_expentries = new ArrayList<int[]>();
  private int[] jj_expentry;
  private int jj_kind = -1;
  private int[] jj_lasttokens = new int[100];
  private int jj_endpos;

  private void jj_add_error_token(int kind, int pos) {
    if (pos >= 100) return;
    if (pos == jj_endpos + 1) {
      jj_lasttokens[jj_endpos++] = kind;
    } else if (jj_endpos != 0) {
      jj_expentry = new int[jj_endpos];
      for (int i = 0; i < jj_endpos; i++) {
        jj_expentry[i] = jj_lasttokens[i];
      }
      jj_entries_loop: for (java.util.Iterator<?> it = jj_expentries.iterator(); it.hasNext();) {
        int[] oldentry = (int[])(it.next());
        if (oldentry.length == jj_expentry.length) {
          for (int i = 0; i < jj_expentry.length; i++) {
            if (oldentry[i] != jj_expentry[i]) {
              continue jj_entries_loop;
            }
          }
          jj_expentries.add(jj_expentry);
          break jj_entries_loop;
        }
      }
      if (pos != 0) jj_lasttokens[(jj_endpos = pos) - 1] = kind;
    }
  }

  /** Generate ParseException. */
  public ParseException generateParseException() {
    jj_expentries.clear();
    boolean[] la1tokens = new boolean[29];
    if (jj_kind >= 0) {
      la1tokens[jj_kind] = true;
      jj_kind = -1;
    }
    for (int i = 0; i < 10; i++) {
      if (jj_la1[i] == jj_gen) {
        for (int j = 0; j < 32; j++) {
          if ((jj_la1_0[i] & (1<<j)) != 0) {
            la1tokens[j] = true;
          }
        }
      }
    }
    for (int i = 0; i < 29; i++) {
      if (la1tokens[i]) {
        jj_expentry = new int[1];
        jj_expentry[0] = i;
        jj_expentries.add(jj_expentry);
      }
    }
    jj_endpos = 0;
    jj_rescan_token();
    jj_add_error_token(0, 0);
    int[][] exptokseq = new int[jj_expentries.size()][];
    for (int i = 0; i < jj_expentries.size(); i++) {
      exptokseq[i] = jj_expentries.get(i);
    }
    return new ParseException(token, exptokseq, tokenImage);
  }

  /** Enable tracing. */
  final public void enable_tracing() {
  }

  /** Disable tracing. */
  final public void disable_tracing() {
  }

  private void jj_rescan_token() {
    jj_rescan = true;
    for (int i = 0; i < 1; i++) {
    try {
      JJCalls p = jj_2_rtns[i];
      do {
        if (p.gen > jj_gen) {
          jj_la = p.arg; jj_lastpos = jj_scanpos = p.first;
          switch (i) {
            case 0: jj_3_1(); break;
          }
        }
        p = p.next;
      } while (p != null);
      } catch(LookaheadSuccess ls) { }
    }
    jj_rescan = false;
  }

  private void jj_save(int index, int xla) {
    JJCalls p = jj_2_rtns[index];
    while (p.gen > jj_gen) {
      if (p.next == null) { p = p.next = new JJCalls(); break; }
      p = p.next;
    }
    p.gen = jj_gen + xla - jj_la; p.first = token; p.arg = xla;
  }

  static final class JJCalls {
    int gen;
    Token first;
    int arg;
    JJCalls next;
  }

}
