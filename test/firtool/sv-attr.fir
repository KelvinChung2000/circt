; RUN: firtool %s | FileCheck %s

FIRRTL version 4.0.0
circuit Foo: %[[{"class": "firrtl.AttributeAnnotation",
                 "description": "keep_hierarchy = \"true\"",
                 "target": "~Foo|Foo"},
                {"class": "firrtl.AttributeAnnotation",
                 "description": "mark_debug = \"yes\"",
                 "target": "~Foo|Foo>w"},
                {"class": "firrtl.transforms.DontTouchAnnotation",
                 "target": "~Foo|Foo>w"},
                {"class": "firrtl.AttributeAnnotation",
                 "description": "mark_debug = \"yes\"",
                 "target": "~Foo|Foo>n"},
                {"class": "firrtl.transforms.DontTouchAnnotation",
                 "target": "~Foo|Foo>n"},
                {"class": "firrtl.AttributeAnnotation",
                 "description": "keep = \"true\"",
                 "target": "~Foo|Foo>r"}]]
  ; CHECK: (* keep_hierarchy = "true" *)
  ; CHECK-NEXT: module Foo
  public module Foo:
    input a: UInt<1>
    input clock: Clock
    output b1: UInt<1>
    output b2: UInt<1>
    ; CHECK: (* mark_debug = "yes" *)
    ; CHECK-NEXT: wire w
    wire w: UInt<1>
    ; CHECK:      (* mark_debug = "yes" *)
    ; CHECK-NEXT: wire n
    node n = w;
    ; CHECK:      (* keep = "true" *)
    ; CHECK-NEXT: reg r
    reg r: UInt<1>, clock
    connect w, a
    connect b1, n
    connect r, a
    connect b2, r
