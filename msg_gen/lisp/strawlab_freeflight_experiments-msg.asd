
(cl:in-package :asdf)

(defsystem "strawlab_freeflight_experiments-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "CylinderGratingInfo" :depends-on ("_package_CylinderGratingInfo"))
    (:file "_package_CylinderGratingInfo" :depends-on ("_package"))
  ))